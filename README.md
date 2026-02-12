# Semantic project navigator

## Usage

Currently I've only built and run this script using Nix.  However, you can feel free to
submit pull requests for other installation instructions if you've vetted them.

No matter what you do you will need to provide an `OPENAI_API_KEY` environment variable in
order to use this script:

```ShellSession
$ export OPENAI_API_KEY="$(< ./path/to/openai.key)"
```

Once you do that you can run the script in a single Nix command, like this:

```ShellSession
$ nix run github:Gabriella439/semantic-navigator -- ./path/to/repository
```

… or you can build the script and run it separately:

```ShellSession
$ nix build github:Gabriella439/semantic-navigator

$ ./result/bin/semantic-navigator ./path/to/repository
```

… or you can install the script:

```ShellSession
$ nix profile install github:Gabriella439/semantic-navigator

$ semantic-navigator ./path/to/repostiory
```

## Development

If you use Nix and `direnv` this project provides a `.envrc` which automatically provides
a virtual environment with all the necessary dependencies (both Python and non-Python
dependencies).

Otherwise if you don't use `direnv` you can enter the virtual environment using:

```ShellSession
$ nix develop
```
