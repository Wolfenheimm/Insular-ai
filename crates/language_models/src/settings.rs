use anyhow::Result;
use gpui::App;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsSources};

use crate::provider::{
    lmstudio::{LmStudioSettings, AvailableModel as LmStudioAvailableModel},
    ollama::{OllamaSettings, AvailableModel as OllamaAvailableModel},
};

/// Initializes the language model settings.
pub fn init(cx: &mut App) {
    AllLanguageModelSettings::register(cx);
}

#[derive(Default)]
pub struct AllLanguageModelSettings {
    pub lmstudio: LmStudioSettings,
    pub ollama: OllamaSettings,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct AllLanguageModelSettingsContent {
    pub lmstudio: Option<LmStudioSettingsContent>,
    pub ollama: Option<OllamaSettingsContent>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct LmStudioSettingsContent {
    pub api_url: Option<String>,
    pub available_models: Option<Vec<LmStudioAvailableModel>>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct OllamaSettingsContent {
    pub api_url: Option<String>,
    pub available_models: Option<Vec<OllamaAvailableModel>>,
}

impl settings::Settings for AllLanguageModelSettings {
    const KEY: Option<&'static str> = Some("language_models");

    const PRESERVED_KEYS: Option<&'static [&'static str]> = Some(&["version"]);

    type FileContent = AllLanguageModelSettingsContent;

    fn load(sources: SettingsSources<Self::FileContent>, _: &mut App) -> Result<Self> {
        fn merge<T>(target: &mut T, value: Option<T>) {
            if let Some(value) = value {
                *target = value;
            }
        }

        let mut settings = AllLanguageModelSettings::default();

        for value in sources.defaults_and_customizations() {
            // LM Studio
            let lmstudio = value.lmstudio.clone();
            merge(
                &mut settings.lmstudio.api_url,
                value.lmstudio.as_ref().and_then(|s| s.api_url.clone()),
            );
            merge(
                &mut settings.lmstudio.available_models,
                lmstudio.as_ref().and_then(|s| s.available_models.clone()),
            );

            // Ollama
            let ollama = value.ollama.clone();
            merge(
                &mut settings.ollama.api_url,
                value.ollama.as_ref().and_then(|s| s.api_url.clone()),
            );
            merge(
                &mut settings.ollama.available_models,
                ollama.as_ref().and_then(|s| s.available_models.clone()),
            );
        }

        if settings.lmstudio.api_url.is_empty() {
            settings.lmstudio.api_url = "http://localhost:1234/api/v0".to_string();
        }

        if settings.ollama.api_url.is_empty() {
            settings.ollama.api_url = "http://localhost:11434/api".to_string();
        }

        Ok(settings)
    }

    fn import_from_vscode(_vscode: &settings::VsCodeSettings, _current: &mut Self::FileContent) {}
}
