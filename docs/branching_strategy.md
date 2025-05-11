# Branching Strategy

## Overview

This document outlines the branching strategy for the IDP-System project to ensure consistent and organized development practices.

## Branch Types

### Main Branches

- **main**: Production-ready code
- **develop**: Integration branch for feature development

### Supporting Branches

- **feature/**: New features (branched from develop)
- **bugfix/**: Bug fixes (branched from develop)
- **hotfix/**: Urgent production fixes (branched from main)
- **release/**: Preparation for a new release (branched from develop)

## Workflow

1. **Feature Development**
   - Create a feature branch from develop
   - Implement and test the feature
   - Create a pull request to merge back to develop
   - After review and approval, merge to develop

2. **Bug Fixes**
   - Create a bugfix branch from develop
   - Fix the bug and test
   - Create a pull request to merge back to develop
   - After review and approval, merge to develop

3. **Hotfixes**
   - Create a hotfix branch from main
   - Fix the critical bug
   - Create pull requests to merge to both main and develop
   - After review and approval, merge to both branches

4. **Releases**
   - Create a release branch from develop
   - Perform final testing and version bumping
   - Create pull requests to merge to both main and develop
   - After review and approval, tag the commit in main with the version number

## Branch Naming Conventions

- Feature branches: `feature/feature-name`
- Bug fix branches: `bugfix/bug-description`
- Hotfix branches: `hotfix/issue-description`
- Release branches: `release/version-number`

## Pull Request Process

1. Create a pull request with a descriptive title and detailed description
2. Ensure all CI checks pass
3. Request review from at least one team member
4. Address all review comments
5. Once approved, merge the pull request
