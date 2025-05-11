# Release Management Process

## Version Numbering

We follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Add functionality in a backward compatible manner
- **PATCH**: Backward compatible bug fixes

## Release Preparation

1. **Create Release Branch**
   ```bash
   git checkout develop
   git pull
   git checkout -b release/X.Y.Z
   ```

2. **Update Version Numbers**
   - Update version in main.py, setup.py, and any other relevant files
   - Update CHANGELOG.md with all changes since the last release

3. **Run Final Tests**
   ```bash
   pytest
   python -m pytest tests/integration
   python -m pytest tests/performance
   ```

4. **Create Pull Request**
   - Create a PR from release/X.Y.Z to main
   - Create a PR from release/X.Y.Z to develop

5. **After PR Approval**
   - Merge the PR to main
   - Merge the PR to develop

## Release Publication

1. **Tag the Release**
   ```bash
   git checkout main
   git pull
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   git push origin vX.Y.Z
   ```

2. **Create GitHub Release**
   - Go to GitHub Releases
   - Create a new release using the tag
   - Include the relevant section from CHANGELOG.md
   - Attach any relevant artifacts

3. **Build and Publish Docker Image**
   ```bash
   docker build -t idp-system:X.Y.Z .
   docker tag idp-system:X.Y.Z registry.example.com/idp-system:X.Y.Z
   docker push registry.example.com/idp-system:X.Y.Z
   ```

4. **Deploy to Production**
   - Follow your deployment procedure
   - Monitor the deployment for any issues

## Hotfix Process

If a critical issue is found in production:

1. Create a hotfix branch from main
2. Fix the issue
3. Create PRs to both main and develop
4. Follow the regular release process, but increment only the PATCH version
