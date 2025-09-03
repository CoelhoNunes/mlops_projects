# Contribution Heartbeat Setup Guide

## Overview
This document explains how to set up the contribution heartbeat system that will create daily commits to maintain GitHub contribution activity.

## What It Does
The contribution heartbeat workflow:
1. **Runs daily** at 17:00 UTC (configurable)
2. **Creates a commit** to the main branch with a timestamp
3. **Uses your identity** (CoelhoNunes) for the commit author
4. **Skips CI** to avoid infinite loops
5. **Only runs when** the required PAT secret is present

## Setup Steps

### 1. Create Fine-Grained Personal Access Token (PAT)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Click "Generate new token"
3. Configure the token:
   - **Token name**: `PAT_FOR_CONTRIB`
   - **Expiration**: Choose appropriate duration (e.g., 90 days)
   - **Repository access**: Select "Only select repositories" → Choose `CoelhoNunes/mlops_projects`
   - **Permissions**:
     - Repository permissions → Contents → Read and write ✅

4. Click "Generate token"
5. **Copy the token** (you won't see it again!)

### 2. Add PAT to Repository Secrets

1. Go to your repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Configure:
   - **Name**: `PAT_FOR_CONTRIB`
   - **Value**: Paste the token from step 1
4. Click "Add secret"

### 3. Verify Setup

1. **Check workflow file**: `.github/workflows/contrib-heartbeat.yml` should exist
2. **Manual test**: Go to Actions tab → "Contribution Heartbeat" → "Run workflow"
3. **Verify commit**: Check that a commit appears on main branch with message "chore(contrib): heartbeat [skip ci]"
4. **Check contributions**: Your GitHub profile should show a green square for today

## How It Works

### Workflow Triggers
- **Manual**: Run workflow button in Actions tab
- **Scheduled**: Daily at 17:00 UTC (configurable in the workflow)

### What Happens
1. Workflow checks if `PAT_FOR_CONTRIB` secret exists
2. If secret exists:
   - Checks out repository
   - Updates `.github/_last_heartbeat` file with UTC timestamp
   - Commits with message "chore(contrib): heartbeat [skip ci]"
   - Pushes to main branch using your PAT
3. If secret missing:
   - Workflow completes successfully without error
   - No commit is made

### Safety Features
- **No infinite loops**: Commit message contains `[skip ci]` which prevents other workflows from running
- **Conditional execution**: Only runs when PAT is present
- **Main branch only**: Never runs on PRs or other branches
- **Concurrency control**: Prevents multiple heartbeat runs simultaneously

## Customization

### Change Schedule
Edit the cron expression in `.github/workflows/contrib-heartbeat.yml`:
```yaml
schedule:
  - cron: "0 17 * * *"  # Daily at 17:00 UTC
```

Common cron patterns:
- `"0 9 * * *"` - Daily at 09:00 UTC
- `"0 12 * * 1-5"` - Weekdays at 12:00 UTC
- `"0 0 1 * *"` - Monthly on 1st at 00:00 UTC

### Change Author Information
Edit the git config in the workflow:
```yaml
- name: Configure author
  run: |
    git config user.name "CoelhoNunes"
    git config user.email "coelhonunes@users.noreply.github.com"
```

### Change Heartbeat File
The workflow creates `.github/_last_heartbeat` by default. You can modify the path and content in the "Update heartbeat file" step.

## Troubleshooting

### Workflow Not Running
1. Check if PAT secret exists in repository settings
2. Verify PAT has correct permissions (Contents: Read & Write)
3. Check workflow file syntax in Actions tab

### Commits Not Appearing
1. Verify PAT has write access to the repository
2. Check workflow logs for authentication errors
3. Ensure workflow is targeting main branch

### CI Still Running
1. Verify commit message contains `[skip ci]`
2. Check that all workflows have the skip condition:
   ```yaml
   if: ${{ !contains(github.event.head_commit.message, '[skip ci]') }}
   ```

### No Green Squares
1. Verify commit author email matches your GitHub account
2. Check that commits are on the default branch (main)
3. Wait up to 24 hours for contributions to update

## Security Notes

- **PAT scope**: Only grant the minimum required permissions (Contents: Read & Write)
- **Repository access**: Limit to only the specific repository
- **Token expiration**: Set appropriate expiration and rotate regularly
- **Secret storage**: Never commit the PAT to the repository

## Benefits

✅ **Maintains activity**: Daily commits keep your contribution graph active
✅ **No manual work**: Fully automated once set up
✅ **Safe operation**: Won't interfere with normal CI/CD
✅ **Configurable**: Easy to adjust schedule and behavior
✅ **Identity preservation**: Commits appear under your name

## Maintenance

- **Rotate PAT**: Update the secret before expiration
- **Monitor logs**: Check workflow runs occasionally
- **Update schedule**: Adjust timing as needed
- **Review commits**: Ensure heartbeat commits are working as expected
