import argostranslate.package
import argostranslate.translate


# Download all languages
def download_all_languages():
    argostranslate.package.update_package_index()

    available_packages = argostranslate.package.get_available_packages()

    for package in available_packages:
        print(
            f'Downloading and installing: {package.from_code} to {package.to_code}')
        argostranslate.package.install_from_path(package.download())


def download_languages_for_demo():
    argostranslate.package.update_package_index()

    available_packages = argostranslate.package.get_available_packages()

    to_codes_we_need = ['es', 'fr', 'ja', 'de',
                        'zh', 'hi', 'fr', 'ar', 'ru', 'ur', 'bn']

    for package in available_packages:
        if package.from_code == 'en' and package.to_code in to_codes_we_need:
            print(
                f'Downloading and installing: {package.from_code} to {package.to_code}')
            argostranslate.package.install_from_path(package.download())

# Translate text


def translate_text(text, to_code):
    installed_languages = argostranslate.translate.get_installed_languages()

    list_of_language_to_codes = []
    for language in installed_languages:
        list_of_language_to_codes.append(language.code)

    if to_code not in list_of_language_to_codes:
        print(f'Language code {to_code} not found in installed languages')
        return

    from_lang = list(filter(
        lambda x: x.code == 'en',
        installed_languages))[0]

    to_lang = list(filter(
        lambda x: x.code == to_code,
        installed_languages))[0]

    translation = from_lang.get_translation(to_lang)

    translatedText = translation.translate(text)

    return translatedText

# Test all languages


def test_all_languages():
    installed_languages = argostranslate.translate.get_installed_languages()

    list_of_language_to_codes = []
    for language in installed_languages:
        list_of_language_to_codes.append(language.code)

    for to_code in list_of_language_to_codes:
        print(f"Testing translation to {to_code}")
        translated_text = translate_text("Hello, how are you?", to_code)
        print(f"Translated text: {translated_text}")


def test_languages_we_selected_for_demo():
    to_codes_we_need = ['es', 'fr', 'ja', 'de',
                        'zh', 'hi', 'fr', 'ar', 'ru', 'ur', 'bn']

    for to_code in to_codes_we_need:
        print(f"Testing translation to {to_code}")
        translated_text = translate_text("Hello, how are you?", to_code)
        print(f"Translated text: {translated_text}")

# Testing
# download_all_languages()

# test_all_languages()


#download_languages_for_demo()

#test_languages_we_selected_for_demo()
