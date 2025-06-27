use anyhow::{Result, anyhow};
use collections::HashMap;
use futures::Stream;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream, future};
use gpui::{AnyView, App, AsyncApp, Context, Subscription, Task};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelToolChoice, LanguageModelToolResultContent, LanguageModelToolUse, MessageContent,
    StopReason, TokenUsage,
};
use language_model::{
    LanguageModel, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, RateLimiter, Role,
};
use lmstudio::{ModelType, get_models};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use std::pin::Pin;
use std::str::FromStr;
use std::{collections::BTreeMap, sync::Arc};
use ui::{ButtonLike, Indicator, List, prelude::*};
use util::ResultExt;

use crate::AllLanguageModelSettings;
use crate::ui::InstructionListItem;

const LMSTUDIO_DOWNLOAD_URL: &str = "https://lmstudio.ai/download";
const LMSTUDIO_CATALOG_URL: &str = "https://lmstudio.ai/models";
const LMSTUDIO_SITE: &str = "https://lmstudio.ai/";

const PROVIDER_ID: &str = "lmstudio";
const PROVIDER_NAME: &str = "LM Studio";

// Required models for insulated AI system - best models for local development
const REQUIRED_MODELS: &[&str] = &[
    "codellama",     // Yi Coder for coding (9B) - CodeLlama alternative
    "codestral",     // Excellent coding model (22B)
    "phi",           // Latest Phi-4 model (14B) 
    "qwen-coder",    // Qwen coding specialist (14B)
    "llama",         // General purpose Llama (8B)
    "phi-mini",      // Efficient Phi with long context (3.8B)
    "qwen-reasoning" // Reasoning specialist (8B)
];

#[derive(Default, Debug, Clone, PartialEq)]
pub struct LmStudioSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AvailableModel {
    pub name: String,
    pub display_name: Option<String>,
    pub max_tokens: u64,
    pub supports_tool_calls: bool,
    pub supports_images: bool,
}

pub struct LmStudioLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: gpui::Entity<State>,
}

pub struct State {
    http_client: Arc<dyn HttpClient>,
    available_models: Vec<lmstudio::Model>,
    fetch_model_task: Option<Task<Result<()>>>,
    _subscription: Subscription,
}

impl State {
    fn is_authenticated(&self) -> bool {
        !self.available_models.is_empty()
    }

    // Check if required models (CodeLlama and Phi-2) are available
    fn has_required_models(&self) -> bool {
        let available_model_names: Vec<String> = self.available_models
            .iter()
            .map(|m| m.name.to_lowercase())
            .collect();
        
        REQUIRED_MODELS.iter().all(|required| {
            available_model_names.iter().any(|available| {
                available.contains(*required)
            })
        })
    }

    // Get missing required models
    fn get_missing_models(&self) -> Vec<&str> {
        let available_model_names: Vec<String> = self.available_models
            .iter()
            .map(|m| m.name.to_lowercase())
            .collect();
        
        REQUIRED_MODELS.iter()
            .filter(|required| {
                !available_model_names.iter().any(|available| {
                    available.contains(**required)
                })
            })
            .copied()
            .collect()
    }

    // Check if LMStudio is running by attempting to fetch models
    fn is_lmstudio_running(&self, cx: &mut Context<Self>) -> Task<Result<bool>> {
        let settings = &AllLanguageModelSettings::get_global(cx).lmstudio;
        let http_client = self.http_client.clone();
        let api_url = settings.api_url.clone();

        cx.spawn(async move |_this, _cx| {
            match get_models(http_client.as_ref(), &api_url, None).await {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        })
    }

    // Try to start LMStudio server
    fn start_lmstudio(&self, cx: &mut Context<Self>) -> Task<Result<()>> {
        cx.spawn(async move |_this, _cx| {
            // Try to start LMStudio server via command line
            let output = std::process::Command::new("lms")
                .args(&["server", "start"])
                .output();

            match output {
                Ok(output) => {
                    if output.status.success() {
                        Ok(())
                    } else {
                        Err(anyhow!("Failed to start LMStudio server: {}", 
                            String::from_utf8_lossy(&output.stderr)))
                    }
                }
                Err(_) => {
                    // If command fails, try to open LMStudio app
                    #[cfg(target_os = "macos")]
                    {
                        let _ = std::process::Command::new("open")
                            .args(&["-a", "LM Studio"])
                            .spawn();
                    }
                    #[cfg(target_os = "windows")]
                    {
                        let _ = std::process::Command::new("cmd")
                            .args(&["/c", "start", "lmstudio"])
                            .spawn();
                    }
                    #[cfg(target_os = "linux")]
                    {
                        let _ = std::process::Command::new("lmstudio")
                            .spawn();
                    }
                    
                    Err(anyhow!("LMStudio command not found. Please install LMStudio and ensure it's in your PATH."))
                }
            }
        })
    }

    // Install a required model
    fn install_model(&self, model_name: &str, cx: &mut Context<Self>) -> Task<Result<()>> {
        let model_name = model_name.to_string();
        cx.spawn(async move |_this, _cx| {
            // Map generic model names to specific downloadable models
            let download_model = match model_name.as_str() {
                "codellama" => "yi-coder-9b",  // Yi Coder is excellent for coding tasks
                "codestral" => "codestral-22b",
                "phi" => "phi-4", 
                "qwen-coder" => "qwen2.5-coder-14b",
                "llama" => "llama-3.1-8b",
                "phi-mini" => "phi-3.1-mini-128k",
                "qwen-reasoning" => "qwen3-8b",
                _ => &model_name,
            };

            log::info!("Attempting to install model: {} ({})", model_name, download_model);
            
            let output = std::process::Command::new("lms")
                .args(&["get", download_model])
                .output();

            match output {
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    
                    if output.status.success() {
                        log::info!("Successfully installed model: {}", download_model);
                        Ok(())
                    } else {
                        let error_msg = format!("Failed to install model {} ({}): {}", 
                            model_name, download_model, stderr);
                        log::error!("{}", error_msg);
                        log::error!("Command output: {}", stdout);
                        Err(anyhow!(error_msg))
                    }
                }
                Err(e) => {
                    let error_msg = format!("LMStudio CLI (lms) not found. Please install LMStudio CLI. Error: {}", e);
                    log::error!("{}", error_msg);
                    Err(anyhow!(error_msg))
                }
            }
        })
    }

    fn fetch_models(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let settings = &AllLanguageModelSettings::get_global(cx).lmstudio;
        let http_client = self.http_client.clone();
        let api_url = settings.api_url.clone();

        // As a proxy for the server being "authenticated", we'll check if its up by fetching the models
        cx.spawn(async move |this, cx| {
            let models = get_models(http_client.as_ref(), &api_url, None).await?;

            let mut models: Vec<lmstudio::Model> = models
                .into_iter()
                .filter(|model| model.r#type != ModelType::Embeddings)
                .map(|model| {
                    lmstudio::Model::new(
                        &model.id,
                        None,
                        model
                            .loaded_context_length
                            .or_else(|| model.max_context_length),
                        model.capabilities.supports_tool_calls(),
                        model.capabilities.supports_images() || model.r#type == ModelType::Vlm,
                    )
                })
                .collect();

            models.sort_by(|a, b| a.name.cmp(&b.name));

            this.update(cx, |this, cx| {
                this.available_models = models;
                cx.notify();
            })
        })
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        let task = self.fetch_models(cx);
        self.fetch_model_task.replace(task);
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        let fetch_models_task = self.fetch_models(cx);
        cx.spawn(async move |_this, _cx| Ok(fetch_models_task.await?))
    }
}

impl LmStudioLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let this = Self {
            http_client: http_client.clone(),
            state: cx.new(|cx| {
                let subscription = cx.observe_global::<SettingsStore>({
                    let mut settings = AllLanguageModelSettings::get_global(cx).lmstudio.clone();
                    move |this: &mut State, cx| {
                        let new_settings = &AllLanguageModelSettings::get_global(cx).lmstudio;
                        if &settings != new_settings {
                            settings = new_settings.clone();
                            this.restart_fetch_models_task(cx);
                            cx.notify();
                        }
                    }
                });

                State {
                    http_client,
                    available_models: Default::default(),
                    fetch_model_task: None,
                    _subscription: subscription,
                }
            }),
        };
        this.state
            .update(cx, |state, cx| state.restart_fetch_models_task(cx));
        this
    }
}

impl LanguageModelProviderState for LmStudioLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<gpui::Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for LmStudioLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn icon(&self) -> IconName {
        IconName::AiLmStudio
    }

    fn default_model(&self, _: &App) -> Option<Arc<dyn LanguageModel>> {
        // We shouldn't try to select default model, because it might lead to a load call for an unloaded model.
        // In a constrained environment where user might not have enough resources it'll be a bad UX to select something
        // to load by default.
        None
    }

    fn default_fast_model(&self, _: &App) -> Option<Arc<dyn LanguageModel>> {
        // See explanation for default_model.
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models: BTreeMap<String, lmstudio::Model> = BTreeMap::default();

        // Add models from the LM Studio API
        for model in self.state.read(cx).available_models.iter() {
            models.insert(model.name.clone(), model.clone());
        }

        // Override with available models from settings
        for model in AllLanguageModelSettings::get_global(cx)
            .lmstudio
            .available_models
            .iter()
        {
            models.insert(
                model.name.clone(),
                lmstudio::Model {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    supports_tool_calls: model.supports_tool_calls,
                    supports_images: model.supports_images,
                },
            );
        }

        models
            .into_values()
            .map(|model| {
                Arc::new(LmStudioLanguageModel {
                    id: LanguageModelId::from(model.name.clone()),
                    model: model.clone(),
                    http_client: self.http_client.clone(),
                    request_limiter: RateLimiter::new(4),
                }) as Arc<dyn LanguageModel>
            })
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(&self, _window: &mut Window, cx: &mut App) -> AnyView {
        let state = self.state.clone();
        cx.new(|cx| ConfigurationView::new(state, cx)).into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| state.fetch_models(cx))
    }
}

pub struct LmStudioLanguageModel {
    id: LanguageModelId,
    model: lmstudio::Model,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl LmStudioLanguageModel {
    fn to_lmstudio_request(
        &self,
        request: LanguageModelRequest,
    ) -> lmstudio::ChatCompletionRequest {
        let mut messages = Vec::new();

        for message in request.messages {
            for content in message.content {
                match content {
                    MessageContent::Text(text) => add_message_content_part(
                        lmstudio::MessagePart::Text { text },
                        message.role,
                        &mut messages,
                    ),
                    MessageContent::Thinking { .. } => {}
                    MessageContent::RedactedThinking(_) => {}
                    MessageContent::Image(image) => {
                        add_message_content_part(
                            lmstudio::MessagePart::Image {
                                image_url: lmstudio::ImageUrl {
                                    url: image.to_base64_url(),
                                    detail: None,
                                },
                            },
                            message.role,
                            &mut messages,
                        );
                    }
                    MessageContent::ToolUse(tool_use) => {
                        let tool_call = lmstudio::ToolCall {
                            id: tool_use.id.to_string(),
                            content: lmstudio::ToolCallContent::Function {
                                function: lmstudio::FunctionContent {
                                    name: tool_use.name.to_string(),
                                    arguments: serde_json::to_string(&tool_use.input)
                                        .unwrap_or_default(),
                                },
                            },
                        };

                        if let Some(lmstudio::ChatMessage::Assistant { tool_calls, .. }) =
                            messages.last_mut()
                        {
                            tool_calls.push(tool_call);
                        } else {
                            messages.push(lmstudio::ChatMessage::Assistant {
                                content: None,
                                tool_calls: vec![tool_call],
                            });
                        }
                    }
                    MessageContent::ToolResult(tool_result) => {
                        let content = match &tool_result.content {
                            LanguageModelToolResultContent::Text(text) => {
                                vec![lmstudio::MessagePart::Text {
                                    text: text.to_string(),
                                }]
                            }
                            LanguageModelToolResultContent::Image(image) => {
                                vec![lmstudio::MessagePart::Image {
                                    image_url: lmstudio::ImageUrl {
                                        url: image.to_base64_url(),
                                        detail: None,
                                    },
                                }]
                            }
                        };

                        messages.push(lmstudio::ChatMessage::Tool {
                            content: content.into(),
                            tool_call_id: tool_result.tool_use_id.to_string(),
                        });
                    }
                }
            }
        }

        lmstudio::ChatCompletionRequest {
            model: self.model.name.clone(),
            messages,
            stream: true,
            max_tokens: Some(-1),
            stop: Some(request.stop),
            // In LM Studio you can configure specific settings you'd like to use for your model.
            // For example Qwen3 is recommended to be used with 0.7 temperature.
            // It would be a bad UX to silently override these settings from Zed, so we pass no temperature as a default.
            temperature: request.temperature.or(None),
            tools: request
                .tools
                .into_iter()
                .map(|tool| lmstudio::ToolDefinition::Function {
                    function: lmstudio::FunctionDefinition {
                        name: tool.name,
                        description: Some(tool.description),
                        parameters: Some(tool.input_schema),
                    },
                })
                .collect(),
            tool_choice: request.tool_choice.map(|choice| match choice {
                LanguageModelToolChoice::Auto => lmstudio::ToolChoice::Auto,
                LanguageModelToolChoice::Any => lmstudio::ToolChoice::Required,
                LanguageModelToolChoice::None => lmstudio::ToolChoice::None,
            }),
        }
    }

    fn stream_completion(
        &self,
        request: lmstudio::ChatCompletionRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<lmstudio::ResponseStreamEvent>>>,
    > {
        let http_client = self.http_client.clone();
        let Ok(api_url) = cx.update(|cx| {
            let settings = &AllLanguageModelSettings::get_global(cx).lmstudio;
            settings.api_url.clone()
        }) else {
            return futures::future::ready(Err(anyhow!("App state dropped"))).boxed();
        };

        let future = self.request_limiter.stream(async move {
            let request = lmstudio::stream_chat_completion(http_client.as_ref(), &api_url, request);
            let response = request.await?;
            Ok(response)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for LmStudioLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn supports_tools(&self) -> bool {
        self.model.supports_tool_calls()
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        self.supports_tools()
            && match choice {
                LanguageModelToolChoice::Auto => true,
                LanguageModelToolChoice::Any => true,
                LanguageModelToolChoice::None => true,
            }
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images
    }

    fn telemetry_id(&self) -> String {
        format!("lmstudio/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        _cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        // Endpoint for this is coming soon. In the meantime, hacky estimation
        let token_count = request
            .messages
            .iter()
            .map(|msg| msg.string_contents().split_whitespace().count())
            .sum::<usize>();

        let estimated_tokens = (token_count as f64 * 0.75) as u64;
        async move { Ok(estimated_tokens) }.boxed()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        let request = self.to_lmstudio_request(request);
        let completions = self.stream_completion(request, cx);
        async move {
            let mapper = LmStudioEventMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}

struct LmStudioEventMapper {
    tool_calls_by_index: HashMap<usize, RawToolCall>,
}

impl LmStudioEventMapper {
    fn new() -> Self {
        Self {
            tool_calls_by_index: HashMap::default(),
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<lmstudio::ResponseStreamEvent>>>>,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events.flat_map(move |event| {
            futures::stream::iter(match event {
                Ok(event) => self.map_event(event),
                Err(error) => vec![Err(LanguageModelCompletionError::Other(anyhow!(error)))],
            })
        })
    }

    pub fn map_event(
        &mut self,
        event: lmstudio::ResponseStreamEvent,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        let Some(choice) = event.choices.into_iter().next() else {
            return vec![Err(LanguageModelCompletionError::Other(anyhow!(
                "Response contained no choices"
            )))];
        };

        let mut events = Vec::new();
        if let Some(content) = choice.delta.content {
            events.push(Ok(LanguageModelCompletionEvent::Text(content)));
        }

        if let Some(reasoning_content) = choice.delta.reasoning_content {
            events.push(Ok(LanguageModelCompletionEvent::Thinking {
                text: reasoning_content,
                signature: None,
            }));
        }

        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                let entry = self.tool_calls_by_index.entry(tool_call.index).or_default();

                if let Some(tool_id) = tool_call.id {
                    entry.id = tool_id;
                }

                if let Some(function) = tool_call.function {
                    if let Some(name) = function.name {
                        // At the time of writing this code LM Studio (0.3.15) is incompatible with the OpenAI API:
                        // 1. It sends function name in the first chunk
                        // 2. It sends empty string in the function name field in all subsequent chunks for arguments
                        // According to https://platform.openai.com/docs/guides/function-calling?api-mode=responses#streaming
                        // function name field should be sent only inside the first chunk.
                        if !name.is_empty() {
                            entry.name = name;
                        }
                    }

                    if let Some(arguments) = function.arguments {
                        entry.arguments.push_str(&arguments);
                    }
                }
            }
        }

        if let Some(usage) = event.usage {
            events.push(Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })));
        }

        match choice.finish_reason.as_deref() {
            Some("stop") => {
                events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
            }
            Some("tool_calls") => {
                events.extend(self.tool_calls_by_index.drain().map(|(_, tool_call)| {
                    match serde_json::Value::from_str(&tool_call.arguments) {
                        Ok(input) => Ok(LanguageModelCompletionEvent::ToolUse(
                            LanguageModelToolUse {
                                id: tool_call.id.into(),
                                name: tool_call.name.into(),
                                is_input_complete: true,
                                input,
                                raw_input: tool_call.arguments,
                            },
                        )),
                        Err(error) => Err(LanguageModelCompletionError::BadInputJson {
                            id: tool_call.id.into(),
                            tool_name: tool_call.name.into(),
                            raw_input: tool_call.arguments.into(),
                            json_parse_error: error.to_string(),
                        }),
                    }
                }));

                events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::ToolUse)));
            }
            Some(stop_reason) => {
                log::error!("Unexpected OpenAI stop_reason: {stop_reason:?}",);
                events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
            }
            None => {}
        }

        events
    }
}

#[derive(Default)]
struct RawToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn add_message_content_part(
    new_part: lmstudio::MessagePart,
    role: Role,
    messages: &mut Vec<lmstudio::ChatMessage>,
) {
    match (role, messages.last_mut()) {
        (Role::User, Some(lmstudio::ChatMessage::User { content }))
        | (
            Role::Assistant,
            Some(lmstudio::ChatMessage::Assistant {
                content: Some(content),
                ..
            }),
        )
        | (Role::System, Some(lmstudio::ChatMessage::System { content })) => {
            content.push_part(new_part);
        }
        _ => {
            messages.push(match role {
                Role::User => lmstudio::ChatMessage::User {
                    content: lmstudio::MessageContent::from(vec![new_part]),
                },
                Role::Assistant => lmstudio::ChatMessage::Assistant {
                    content: Some(lmstudio::MessageContent::from(vec![new_part])),
                    tool_calls: Vec::new(),
                },
                Role::System => lmstudio::ChatMessage::System {
                    content: lmstudio::MessageContent::from(vec![new_part]),
                },
            });
        }
    }
}

struct ConfigurationView {
    state: gpui::Entity<State>,
    loading_models_task: Option<Task<()>>,
}

impl ConfigurationView {
    pub fn new(state: gpui::Entity<State>, cx: &mut Context<Self>) -> Self {
        let loading_models_task = Some(cx.spawn({
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = state
                    .update(cx, |state, cx| state.authenticate(cx))
                    .log_err()
                {
                    task.await.log_err();
                }
                this.update(cx, |this, cx| {
                    this.loading_models_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            state,
            loading_models_task,
        }
    }

    fn retry_connection(&self, cx: &mut App) {
        self.state
            .update(cx, |state, cx| state.fetch_models(cx))
            .detach_and_log_err(cx);
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_authenticated = self.state.read(cx).is_authenticated();
        let state = self.state.read(cx);
        let has_required_models = state.has_required_models();
        let missing_models = state.get_missing_models();
        let state_handle = self.state.clone();

        let lmstudio_intro = "Insulated AI powered by LM Studio - No data sent to external services.";

        if self.loading_models_task.is_some() {
            div().child(Label::new("Loading models...")).into_any()
        } else {
            v_flex()
                .gap_2()
                .child(
                    v_flex().gap_1().child(Label::new(lmstudio_intro)).child(
                        List::new()
                            .child(InstructionListItem::text_only(
                                "LM Studio must be running with required models for insulated AI.",
                            ))
                            .child(InstructionListItem::text_only(
                                "Required models: Yi-Coder (9B), Codestral (22B), Phi-4 (14B), Qwen2.5-Coder (14B), Llama-3.1 (8B), Phi-Mini (3.8B), Qwen3 (8B)",
                            ))
                            .when(!is_authenticated, |this| {
                                this.child(InstructionListItem::text_only(
                                    "LM Studio appears to be offline. Click 'Start LM Studio' below.",
                                ))
                            })
                            .when(is_authenticated && !has_required_models, |this| {
                                this.child(InstructionListItem::text_only(
                                    format!("Missing {} of {} required models: {}. Click 'Install Missing Models' below.", 
                                        missing_models.len(), REQUIRED_MODELS.len(), missing_models.join(", ")),
                                ))
                            }),
                    ),
                )
                .child(
                    h_flex()
                        .w_full()
                        .justify_between()
                        .gap_2()
                        .child(
                            h_flex()
                                .w_full()
                                .gap_2()
                                .child(
                                    Button::new("lmstudio-site", "LM Studio")
                                        .style(ButtonStyle::Subtle)
                                        .icon(IconName::ArrowUpRight)
                                        .icon_size(IconSize::XSmall)
                                        .icon_color(Color::Muted)
                                        .on_click(move |_, _window, cx| {
                                            cx.open_url(LMSTUDIO_SITE)
                                        }),
                                )
                                .when(!is_authenticated, |this| {
                                    this.child(
                                        Button::new(
                                            "download_lmstudio_button",
                                            "Download LM Studio",
                                        )
                                        .style(ButtonStyle::Subtle)
                                        .icon(IconName::ArrowUpRight)
                                        .icon_size(IconSize::XSmall)
                                        .icon_color(Color::Muted)
                                        .on_click(move |_, _window, cx| {
                                            cx.open_url(LMSTUDIO_DOWNLOAD_URL)
                                        }),
                                    )
                                })
                                .child(
                                    Button::new("view-models", "Model Catalog")
                                        .style(ButtonStyle::Subtle)
                                        .icon(IconName::ArrowUpRight)
                                        .icon_size(IconSize::XSmall)
                                        .icon_color(Color::Muted)
                                        .on_click(move |_, _window, cx| {
                                            cx.open_url(LMSTUDIO_CATALOG_URL)
                                        }),
                                ),
                        )
                        .child(
                            h_flex()
                                .gap_2()
                                .when(!is_authenticated, |this| {
                                    this.child(
                                        Button::new("start_lmstudio", "Start LM Studio")
                                            .icon_position(IconPosition::Start)
                                            .icon_size(IconSize::XSmall)
                                            .icon(IconName::Play)
                                            .on_click({
                                                let state = state_handle.clone();
                                                move |_, _window, cx| {
                                                    // Just trigger a simple retry connection which will attempt to start LMStudio
                                                    state
                                                        .update(cx, |state, cx| state.fetch_models(cx))
                                                        .detach_and_log_err(cx);
                                                }
                                            }),
                                    )
                                })
                                .when(is_authenticated && !has_required_models, |this| {
                                    this.child(
                                        Button::new("install_models", "Install Missing Models")
                                            .icon_position(IconPosition::Start)
                                            .icon_size(IconSize::XSmall)
                                            .icon(IconName::Download)
                                            .on_click({
                                                let state = state_handle.clone();
                                                move |_, _window, cx| {
                                                    let missing_models: Vec<String> = state.read(cx).get_missing_models().iter().map(|s| s.to_string()).collect();
                                                    let model_count = missing_models.len();
                                                    
                                                    if model_count == 0 {
                                                        return;
                                                    }
                                                    
                                                    log::info!("Starting parallel installation of {} models: {:?}", model_count, missing_models);
                                                    
                                                    // Install all missing models in parallel
                                                    let install_tasks: Vec<_> = missing_models.into_iter().map(|model_name| {
                                                        state.update(cx, |state, cx| {
                                                            state.install_model(&model_name, cx)
                                                        })
                                                    }).collect();
                                                    
                                                    // Run all installations concurrently
                                                    cx.spawn({
                                                        let state = state.clone();
                                                        async move |cx| {
                                                            log::info!("Running {} model installations in parallel", install_tasks.len());
                                                            
                                                            // Wait for all installations to complete
                                                            let results = future::join_all(install_tasks).await;
                                                            
                                                            // Log results
                                                            let mut success_count = 0;
                                                            let mut failure_count = 0;
                                                            for result in results {
                                                                match result {
                                                                    Ok(_) => success_count += 1,
                                                                    Err(e) => {
                                                                        failure_count += 1;
                                                                        log::error!("Model installation failed: {}", e);
                                                                    }
                                                                }
                                                            }
                                                            
                                                            log::info!("Model installation complete: {} successful, {} failed", success_count, failure_count);
                                                            
                                                            // Refresh the model list after all installations complete
                                                            state
                                                                .update(cx, |state, cx| state.fetch_models(cx))
                                                                .log_err();
                                                        }
                                                    }).detach();
                                                }
                                            }),
                                    )
                                })
                                .when(is_authenticated, |this| {
                                    this.child(
                                        Button::new("retry_lmstudio_models", "Refresh")
                                            .icon_position(IconPosition::Start)
                                            .icon_size(IconSize::XSmall)
                                            .icon(IconName::RotateCcw)
                                            .on_click({
                                                let state = state_handle.clone();
                                                move |_, _window, cx| {
                                                    state
                                                        .update(cx, |state, cx| state.fetch_models(cx))
                                                        .detach_and_log_err(cx);
                                                }
                                            }),
                                    )
                                })
                                .when(is_authenticated && has_required_models, |this| {
                                    this.child(
                                        ButtonLike::new("ready")
                                            .disabled(true)
                                            .cursor_style(gpui::CursorStyle::Arrow)
                                            .child(
                                                h_flex()
                                                    .gap_2()
                                                    .child(Indicator::dot().color(Color::Success))
                                                    .child(Label::new("Ready for Insulated AI"))
                                                    .into_any_element(),
                                            ),
                                    )
                                }),
                        ),
                )
                .into_any()
        }
    }
}
