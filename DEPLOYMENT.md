# Streamlit Cloud Deployment Guide

This guide will help you deploy the Swing Trading Dashboard to Streamlit Cloud for free mobile access.

## Quick Start

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `app.py`
   - Click "Deploy"

3. **Access from Mobile**
   - Your app will be available at: `https://your-app-name.streamlit.app`
   - Open this URL on any mobile device
   - The app is fully responsive and optimized for mobile

## What's Configured

✅ **requirements.txt** - All dependencies listed  
✅ **.streamlit/config.toml** - Mobile-optimized configuration  
✅ **config.py** - Works with Streamlit Cloud's file system  
✅ **Mobile CSS** - Responsive design for all screen sizes  
✅ **Database** - SQLite configured for ephemeral storage (works fine for caching)

## Important Notes

### Database Behavior
- SQLite database is **ephemeral** on Streamlit Cloud (resets on restart)
- This is **acceptable** because:
  - The database is used for caching only
  - The app will fetch fresh data from yfinance when cache is empty
  - First scan after restart will take longer (fetching all data)
  - Subsequent scans will be fast (using cache)

### Mobile Optimization
- Responsive CSS automatically adapts to mobile screens
- Charts are optimized for mobile viewing
- Buttons and inputs are touch-friendly
- Text sizes adjust for readability

### Performance
- First scan: ~2-5 minutes (fetching data for 500 stocks)
- Cached scans: ~30-60 seconds
- After restart: First scan will be slower (cache cleared)

## Troubleshooting

### App won't start
- Check that `requirements.txt` exists and has all dependencies
- Verify `app.py` is in the root directory
- Check Streamlit Cloud logs for errors

### Slow performance
- This is normal for the first scan (fetching 500 stocks)
- Use "Hard Refresh" sparingly (it fetches fresh data)
- Subsequent scans use cache and are much faster

### Database errors
- Database is ephemeral - this is expected behavior
- App will recreate database automatically on restart
- No action needed

## Free Tier Limits

Streamlit Cloud free tier includes:
- ✅ Unlimited apps
- ✅ Public apps (accessible via URL)
- ✅ Mobile access
- ✅ Automatic deployments from GitHub
- ⚠️ Apps sleep after 7 days of inactivity (wake up on first access)

## Support

For Streamlit Cloud issues, visit: https://docs.streamlit.io/streamlit-community-cloud

