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

def translate_text(text, to_code):
    # Translate
    translatedText = argostranslate.translate.translate(text, "en", to_code)

    return translatedText

# print(translate_text("Hello, how are you?", "es"))