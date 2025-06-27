use std::sync::Arc;

use client::{Client, UserStore};
use gpui::{App, Context, Entity};
use language_model::LanguageModelRegistry;

mod provider;
mod settings;
mod ui;

use crate::provider::lmstudio::LmStudioLanguageModelProvider;
use crate::provider::ollama::OllamaLanguageModelProvider;
pub use crate::settings::*;

pub fn init(user_store: Entity<UserStore>, client: Arc<Client>, cx: &mut App) {
    crate::settings::init(cx);
    let registry = LanguageModelRegistry::global(cx);
    registry.update(cx, |registry, cx| {
        register_language_model_providers(registry, user_store, client, cx);
    });
}

fn register_language_model_providers(
    registry: &mut LanguageModelRegistry,
    _user_store: Entity<UserStore>,
    client: Arc<Client>,
    cx: &mut Context<LanguageModelRegistry>,
) {
    // Register LMStudio provider with enhanced insulated AI features
    registry.register_provider(
        LmStudioLanguageModelProvider::new(client.http_client(), cx),
        cx,
    );

    // Register Ollama provider for local AI
    registry.register_provider(
        OllamaLanguageModelProvider::new(client.http_client(), cx),
        cx,
    );
}
