# Git Setup Instructions

## ✅ Completed
- Git repository initialized
- All files committed
- Ready to push to GitHub

## Next Steps

### 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `indian-swing-trading-dashboard` (or your preferred name)
3. Description: "Swing Trading Dashboard for Nifty 500 - Deployed on Streamlit Cloud"
4. Choose Public or Private
5. **DO NOT** check "Initialize with README" (we already have files)
6. Click "Create repository"

### 2. Add Remote and Push

After creating the repository, run these commands (replace `YOUR_USERNAME` and `REPO_NAME` with your actual values):

```bash
cd "d:\Python Projects\Indian Swing Trading with Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### 3. Deploy to Streamlit Cloud

After pushing to GitHub:

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file path: `app.py`
6. Click "Deploy"

Your app will be available at: `https://your-app-name.streamlit.app`

## Current Git Status

- ✅ Repository initialized
- ✅ Initial commit created
- ⏳ Waiting for GitHub repository URL to add remote

