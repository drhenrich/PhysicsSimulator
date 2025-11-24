# Minimal i18n helper for DE/EN
TRANSLATIONS = {
    "title": {"de": "🔬 Physik-Simulator", "en": "🔬 Physics Simulator"},
    "subtitle": {"de": "Interaktive Simulationen für Physikausbildung", "en": "Interactive Simulations for Physics Education"},
    "mechanics": {"de": "🚀 Mechanik & Himmelsmechanik", "en": "🚀 Mechanics & Celestial Mechanics"},
    "optics": {"de": "🔬 Optik", "en": "🔬 Optics"},
    "xray_ct_classic": {"de": "🩻 Xray/CT", "en": "🩻 Xray/CT"},
    "mri_imaging": {"de": "🧲 MRI & Bloch", "en": "🧲 MRI & Bloch"},
    "electromagnetism": {"de": "⚡ Elektromagnetismus", "en": "⚡ Electromagnetism"},
    # Generic labels
    "adv_mech_title": {"de": "⚙️ Erweiterte Mechanik-Presets (3D)", "en": "⚙️ Advanced Mechanics Presets (3D)"},
    "adv_preset_select": {"de": "Preset wählen (erweitert)", "en": "Select preset (advanced)"},
    "restitution": {"de": "Restitution", "en": "Restitution"},
    "drag": {"de": "Luftwiderstand", "en": "Drag"},
    "adv_run": {"de": "▶️ Erweitertes Preset simulieren", "en": "▶️ Run advanced preset"},
    "adv_success": {"de": "✅ {preset} simuliert — {note}", "en": "✅ {preset} simulated — {note}"},
    "choose_preset_warning": {"de": "Bitte ein Preset auswählen.", "en": "Please select a preset."},
    "ct_classic_title": {"de": "🏥 CT-Parameter (Klassisch)", "en": "🏥 CT parameters (classic)"},
    "ct_reduced_title": {"de": "🏥 CT-Parameter (Reduziert)", "en": "🏥 CT parameters (reduced)"},
    "bloch_title": {"de": "🧲 Bloch-Parameter", "en": "🧲 Bloch parameters"},
}


def get_text(key: str, language: str = "de") -> str:
    val = TRANSLATIONS.get(key)
    if val:
        return val.get(language) or val.get("de") or key
    return key


def get_language_name(lang_code: str) -> str:
    return "🇩🇪 Deutsch" if lang_code == "de" else "🇬🇧 English"
