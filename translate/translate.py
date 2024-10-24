import argostranslate.package
import argostranslate.translate


def download_all_languages():
    # Update the package index to get the latest available language packages
    argostranslate.package.update_package_index()

    # Get all available packages
    available_packages = argostranslate.package.get_available_packages()

    # Loop through and install each package
    for package in available_packages:
        print(
            f"Downloading and installing: {package.from_code} to {package.to_code}")
        argostranslate.package.install_from_path(package.download())


def translate_text(text, language):
    # Get the to_code for Albanian (sq), Arabic (ar), Azerbaijani (az), Basque (eu), Bengali (bn), Bulgarian (bg), Catalan (ca), Simplified Chinese (zh), Traditional Chinese (zt), Czech (cs), Danish (da), Dutch (nl), English (en), Esperanto (eo), Estonian (et), Finnish (fi), French (fr), Galician (gl), German (de), Greek (el), Hebrew (he), Hindi (hi), Hungarian (hu), Indonesian (id), Irish (ga), Italian (it), Japanese (ja), Korean (ko), Latvian (lv), Lithuanian (lt), Malay (ms), Norwegian Bokmål (nb), Persian (fa), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Slovak (sk), Slovenian (sl), Spanish (es), Swedish (sv), Tagalog (tl), Thai (th), Turkish (tr), Ukrainian (uk), Urdu (ur).
    to_code = ""

    if language == "albanian":
        to_code = "sq"
    elif language == "arabic":
        to_code = "ar"
    elif language == "azerbaijani":
        to_code = "az"
    elif language == "basque":
        to_code = "eu"
    elif language == "bengali":
        to_code = "bn"
    elif language == "bulgarian":
        to_code = "bg"
    elif language == "catalan":
        to_code = "ca"
    elif language == "simplified chinese":
        to_code = "zh"
    elif language == "traditional chinese":
        to_code = "zt"
    elif language == "czech":
        to_code = "cs"
    elif language == "danish":
        to_code = "da"
    elif language == "dutch":
        to_code = "nl"
    elif language == "english":
        to_code = "en"
    elif language == "esperanto":
        to_code = "eo"
    elif language == "estonian":
        to_code = "et"
    elif language == "finnish":
        to_code = "fi"
    elif language == "french":
        to_code = "fr"
    elif language == "galician":
        to_code = "gl"
    elif language == "german":
        to_code = "de"
    elif language == "greek":
        to_code = "el"
    elif language == "hebrew":
        to_code = "he"
    elif language == "hindi":
        to_code = "hi"
    elif language == "hungarian":
        to_code = "hu"
    elif language == "indonesian":
        to_code = "id"
    elif language == "irish":
        to_code = "ga"
    elif language == "italian":
        to_code = "it"
    elif language == "japanese":
        to_code = "ja"
    elif language == "korean":
        to_code = "ko"
    elif language == "latvian":
        to_code = "lv"
    elif language == "lithuanian":
        to_code = "lt"
    elif language == "malay":
        to_code = "ms"
    elif language == "norwegian bokmal":
        to_code = "nb"
    elif language == "persian":
        to_code = "fa"
    elif language == "polish":
        to_code = "pl"
    elif language == "portuguese":
        to_code = "pt"
    elif language == "romanian":
        to_code = "ro"
    elif language == "russian":
        to_code = "ru"
    elif language == "slovak":
        to_code = "sk"
    elif language == "slovenian":
        to_code = "sl"
    elif language == "spanish":
        to_code = "es"
    elif language == "swedish":
        to_code = "sv"
    elif language == "tagalog":
        to_code = "tl"
    elif language == "thai":
        to_code = "th"
    elif language == "turkish":
        to_code = "tr"
    elif language == "ukrainian":
        to_code = "uk"
    elif language == "urdu":
        to_code = "ur"
    else:
        print("Language not supported")
        return


    # Translate
    translatedText = argostranslate.translate.translate(text, "en", to_code)

    return translatedText

# Test all languages
def test_all_languages():
    languages = [
        "albanian",
        "arabic",
        "azerbaijani",
        "basque",
        "bengali",
        "bulgarian",
        "catalan",
        "simplified chinese",
        "traditional chinese",
        "czech",
        "danish",
        "dutch",
        "english",
        "esperanto",
        "estonian",
        "finnish",
        "french",
        "galician",
        "german",
        "greek",
        "hebrew",
        "hindi",
        "hungarian",
        "indonesian",
        "irish",
        "italian",
        "japanese",
        "korean",
        "latvian",
        "lithuanian",
        "malay",
        "norwegian bokmal",
        "persian",
        "polish",
        "portuguese",
        "romanian",
        "russian",
        "slovak",
        "slovenian",
        "spanish",
        "swedish",
        "tagalog",
        "thai",
        "turkish",
        "ukrainian",
        "urdu"
    ]

    for language in languages:
        print(f"Testing translation to {language}")
        translated_text = translate_text("Hello, how are you?", language)
        print(f"Translated text: {translated_text}")


# First download and install all language packages
download_all_languages()

test_all_languages()
