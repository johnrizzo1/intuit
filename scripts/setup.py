from setuptools import setup, find_packages

setup(
    name="intuit",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai",
        "google-api-python-client",
        "google-auth-oauthlib",
        "requests",
        "numpy",
        "sounddevice",
        "whisper",
        "chromadb",
        "prompt-toolkit",
    ],
    python_requires=">=3.8",
) 