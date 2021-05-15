# How to release

- Make an entry in the CHANGELOG.md, moving changes from UNRELEASED to the version to be created
  if UNRELEASED is up to date, otherwise writing them yourself
- Either tag the commit with the version number via `git tag {VERSION}` and push with `git push
  origin {VERSION}`, or create a release in GitHub with the desired version number (which will
  create the tag for you)
  - The version tag will then trigger the GitHub Action `release.yaml` to package and release on
    PyPI
