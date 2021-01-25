import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
        name="chatbot",
        version="0.0.1",
        author="insublee",
        author_email="643jason@gmail.com",
        description="Memory augmented reinforce learning chatbot",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/insublee/chatbot",
        packages=setuptools.find_packages(),
        classifiers=[
            "Development Status :: 1 - Planning",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
        python_requires='>=3.6',
)

