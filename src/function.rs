use crate::{
    config::{Agent, Config, GlobalConfig},
    utils::*,
};

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[cfg(windows)]
const PATH_SEP: &str = ";";
#[cfg(not(windows))]
const PATH_SEP: &str = ":";

pub fn eval_tool_calls(config: &GlobalConfig, mut calls: Vec<ToolCall>) -> Result<Vec<ToolResult>> {
    let mut output = vec![];
    if calls.is_empty() {
        return Ok(output);
    }
    calls = ToolCall::dedup(calls);
    if calls.is_empty() {
        bail!("The request was aborted because an infinite loop of function calls was detected.")
    }
    let mut is_all_null = true;
    for call in calls {
        let mut result = call.eval(config)?;
        if result.is_null() {
            result = json!("DONE");
        } else {
            is_all_null = false;
        }
        output.push(ToolResult::new(call, result));
    }
    if is_all_null {
        output = vec![];
    }
    Ok(output)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolResult {
    pub call: ToolCall,
    pub output: Value,
}

impl ToolResult {
    pub fn new(call: ToolCall, output: Value) -> Self {
        Self { call, output }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Functions {
    declarations: Vec<FunctionDeclaration>,
}

impl Functions {
    pub fn init(declarations_path: &Path) -> Result<Self> {
        let declarations: Vec<FunctionDeclaration> = if declarations_path.exists() {
            let ctx = || {
                format!(
                    "Failed to load functions at {}",
                    declarations_path.display()
                )
            };
            let content = fs::read_to_string(declarations_path).with_context(ctx)?;
            serde_json::from_str(&content).with_context(ctx)?
        } else {
            vec![]
        };

        Ok(Self { declarations })
    }

    pub fn find(&self, name: &str) -> Option<&FunctionDeclaration> {
        self.declarations.iter().find(|v| v.name == name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.declarations.iter().any(|v| v.name == name)
    }

    pub fn declarations(&self) -> &[FunctionDeclaration] {
        &self.declarations
    }

    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: JsonSchema,
    #[serde(skip_serializing, default)]
    pub agent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,
    #[serde(rename = "anyOf", skip_serializing_if = "Option::is_none")]
    pub any_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_value: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl JsonSchema {
    pub fn is_empty_properties(&self) -> bool {
        match &self.properties {
            Some(v) => v.is_empty(),
            None => true,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<Value>,
}

type CallConfig = (String, String, Vec<String>, HashMap<String, String>);

impl ToolCall {
    pub fn dedup(calls: Vec<Self>) -> Vec<Self> {
        let mut new_calls = vec![];
        let mut seen_ids = HashSet::new();

        for call in calls.into_iter().rev() {
            if let Some(id) = &call.id {
                if !seen_ids.contains(id) {
                    seen_ids.insert(id.clone());
                    new_calls.push(call);
                }
            } else {
                new_calls.push(call);
            }
        }

        new_calls.reverse();
        new_calls
    }

    pub fn new(name: String, arguments: Value, id: Option<String>) -> Self {
        Self {
            name,
            arguments,
            id,
            extra_content: None,
        }
    }

    pub fn eval(&self, config: &GlobalConfig) -> Result<Value> {
        let (call_name, cmd_name, mut cmd_args, envs) = match &config.read().agent {
            Some(agent) => self.extract_call_config_from_agent(config, agent)?,
            None => self.extract_call_config_from_config(config)?,
        };

        let json_data = if self.arguments.is_object() {
            self.arguments.clone()
        } else if let Some(arguments) = self.arguments.as_str() {
            let arguments: Value = serde_json::from_str(arguments).map_err(|_| {
                anyhow!("The call '{call_name}' has invalid arguments: {arguments}")
            })?;
            arguments
        } else {
            bail!(
                "The call '{call_name}' has invalid arguments: {}",
                self.arguments
            );
        };

        cmd_args.push(json_data.to_string());

        let output = match run_llm_function(cmd_name, cmd_args, envs)? {
            Some(contents) => serde_json::from_str(&contents)
                .ok()
                .unwrap_or_else(|| json!({"output": contents})),
            None => Value::Null,
        };

        Ok(output)
    }

    fn extract_call_config_from_agent(
        &self,
        config: &GlobalConfig,
        agent: &Agent,
    ) -> Result<CallConfig> {
        let function_name = self.name.clone();
        match agent.functions().find(&function_name) {
            Some(function) => {
                let agent_name = agent.name().to_string();
                if function.agent {
                    Ok((
                        format!("{agent_name}-{function_name}"),
                        agent_name,
                        vec![function_name],
                        agent.variable_envs(),
                    ))
                } else {
                    Ok((
                        function_name.clone(),
                        function_name,
                        vec![],
                        Default::default(),
                    ))
                }
            }
            None => self.extract_call_config_from_config(config),
        }
    }

    fn extract_call_config_from_config(&self, config: &GlobalConfig) -> Result<CallConfig> {
        let function_name = self.name.clone();
        match config.read().functions.contains(&function_name) {
            true => Ok((
                function_name.clone(),
                function_name,
                vec![],
                Default::default(),
            )),
            false => bail!("Unexpected call: {function_name} {}", self.arguments),
        }
    }
}

pub fn run_llm_function(
    cmd_name: String,
    cmd_args: Vec<String>,
    mut envs: HashMap<String, String>,
) -> Result<Option<String>> {
    let prompt = format!("Call {cmd_name} {}", cmd_args.join(" "));

    let mut bin_dirs: Vec<PathBuf> = vec![];
    if cmd_args.len() > 1 {
        let dir = Config::agent_functions_dir(&cmd_name).join("bin");
        if dir.exists() {
            bin_dirs.push(dir);
        }
    }
    bin_dirs.push(Config::functions_bin_dir());
    let current_path = std::env::var("PATH").context("No PATH environment variable")?;
    let prepend_path = bin_dirs
        .iter()
        .map(|v| format!("{}{PATH_SEP}", v.display()))
        .collect::<Vec<_>>()
        .join("");
    envs.insert("PATH".into(), format!("{prepend_path}{current_path}"));

    let temp_file = temp_file("-eval-", "");
    envs.insert("LLM_OUTPUT".into(), temp_file.display().to_string());

    #[cfg(windows)]
    let cmd_name = polyfill_cmd_name(&cmd_name, &bin_dirs);
    if *IS_STDOUT_TERMINAL {
        println!("{}", dimmed_text(&prompt));
    }
    let execution = run_command(&cmd_name, &cmd_args, Some(envs))
        .map_err(|err| anyhow!("Unable to run {cmd_name}, {err}"));
    let output = if temp_file.exists() {
        fs::read_to_string(&temp_file)
            .context("Failed to retrieve tool call output")
            .map(|contents| (!contents.is_empty()).then_some(contents))
    } else {
        Ok(None)
    };

    if let Ok(Some(contents)) = &output {
        if is_cancelled_tool_output(Some(contents)) {
            bail!("Tool call was cancelled");
        }
    }

    match execution {
        Ok(0) => output,
        Ok(exit_code) => {
            let (output, output_error) = match output {
                Ok(output) => (output, None),
                Err(err) => (None, Some(err.to_string())),
            };
            let error = match output_error {
                Some(err) => format!("Tool call exited with {exit_code}; {err}"),
                None => format!("Tool call exited with {exit_code}"),
            };
            Ok(Some(tool_failure_result(error, Some(exit_code), output)))
        }
        Err(err) => {
            let (output, output_error) = match output {
                Ok(output) => (output, None),
                Err(output_err) => (None, Some(output_err.to_string())),
            };
            let error = match output_error {
                Some(output_err) => format!("{err}; {output_err}"),
                None => err.to_string(),
            };
            Ok(Some(tool_failure_result(error, None, output)))
        }
    }
}

fn is_cancelled_tool_output(output: Option<&str>) -> bool {
    output.is_some_and(|output| {
        output
            .lines()
            .any(|line| line.starts_with("CANCELLED ") || line.starts_with("CANCELLED:"))
    })
}

fn tool_failure_result(error: String, exit_code: Option<i32>, output: Option<String>) -> String {
    let mut result = json!({ "error": error });
    if let Some(exit_code) = exit_code {
        result["exit_code"] = json!(exit_code);
    }
    if let Some(output) = output {
        result["output"] = json!(output);
    }
    result.to_string()
}

#[cfg(windows)]
fn polyfill_cmd_name<T: AsRef<Path>>(cmd_name: &str, bin_dir: &[T]) -> String {
    let cmd_name = cmd_name.to_string();
    if let Ok(exts) = std::env::var("PATHEXT") {
        for name in exts.split(';').map(|ext| format!("{cmd_name}{ext}")) {
            for dir in bin_dir {
                let path = dir.as_ref().join(&name);
                if path.exists() {
                    return name.to_string();
                }
            }
        }
    }
    cmd_name
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt};

    fn create_tool(contents: &str) -> (PathBuf, PathBuf) {
        let dir = temp_file("-function-test-", "");
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("tool.sh");
        fs::write(&script, format!("#!/bin/sh\n{contents}\n")).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        (dir, script)
    }

    fn run_tool(script: &Path) -> Result<Option<String>> {
        run_llm_function(
            script.to_string_lossy().into_owned(),
            vec![],
            Default::default(),
        )
    }

    #[test]
    fn nonzero_tool_exit_returns_structured_result_and_output() {
        let (dir, script) = create_tool("printf 'partial result\\n' >> \"$LLM_OUTPUT\"; exit 7");
        let output = run_tool(&script).unwrap().unwrap();
        let result: Value = serde_json::from_str(&output).unwrap();

        assert_eq!(result["error"], "Tool call exited with 7");
        assert_eq!(result["exit_code"], 7);
        assert_eq!(result["output"], "partial result\n");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ordinary_exit_two_is_not_treated_as_cancellation() {
        let (dir, script) = create_tool("printf 'ordinary failure\\n' >> \"$LLM_OUTPUT\"; exit 2");
        let output = run_tool(&script).unwrap().unwrap();
        let result: Value = serde_json::from_str(&output).unwrap();

        assert_eq!(result["exit_code"], 2);
        assert_eq!(result["output"], "ordinary failure\n");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn explicit_cancellation_remains_an_abort() {
        let (dir, script) =
            create_tool("printf 'CANCELLED action-id /tmp/target\\n' >> \"$LLM_OUTPUT\"; exit 2");
        let error = run_tool(&script).unwrap_err();

        assert_eq!(error.to_string(), "Tool call was cancelled");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cancellation_marker_aborts_even_when_tool_exits_successfully() {
        let (dir, script) =
            create_tool("printf 'CANCELLED action-id /tmp/target\\n' >> \"$LLM_OUTPUT\"; exit 0");
        let error = run_tool(&script).unwrap_err();

        assert_eq!(error.to_string(), "Tool call was cancelled");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn command_start_failure_returns_structured_result() {
        let dir = temp_file("-function-test-missing-", "");
        fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("missing-tool");
        let output = run_tool(&missing).unwrap().unwrap();
        let result: Value = serde_json::from_str(&output).unwrap();

        assert!(result["error"].as_str().unwrap().contains("Unable to run"));
        assert!(result.get("exit_code").is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn successful_tool_output_is_unchanged() {
        let (dir, script) = create_tool("printf 'success\\n' >> \"$LLM_OUTPUT\"");
        assert_eq!(run_tool(&script).unwrap().as_deref(), Some("success\n"));
        fs::remove_dir_all(dir).unwrap();
    }
}
