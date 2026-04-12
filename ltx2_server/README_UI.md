# LTX-2 Web UI

Modern, responsive web interface for LTX-2 video generation.

## Features

### 🎨 User Interface
- **Dark Theme**: Optimized for long generation sessions
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Live progress without page refresh
- **Drag & Drop**: Easy file uploads with preview

### 🎬 Generation Controls
- **Text Prompt**: Primary input for video generation
- **Negative Prompt**: Exclude unwanted elements
- **Image-to-Video**: Upload start/end frames
- **Audio Guide**: Upload audio for synchronized generation
- **Video Settings**: Width, height, frames, FPS, steps, guidance
- **Seed Control**: Reproducible generations
- **Advanced Options**: Attention mode, sliding window

### 📊 Task Management
- **Live Progress**: Real-time step-by-step progress
- **Video Preview**: Watch generated videos in browser
- **Download**: One-click video download
- **Task History**: Browse recent generations
- **Error Handling**: Clear error messages

## Accessing the UI

Once the server is running, access the UI at:

```
http://localhost:8000
```

The API documentation is still available at:
```
http://localhost:8000/docs  (Swagger UI)
http://localhost:8000/redoc  (ReDoc)
```

## Usage Guide

### 1. Basic Generation

1. **Enter Prompt**: Describe the video you want
2. **Adjust Settings** (optional):
   - Width/Height: Must be divisible by 64
   - Frames: Use 17 + 8*n (e.g., 49, 121, 241)
   - Steps: 30-50 for good quality
   - Guidance: 3.5-4.5 for balanced results
3. **Click Generate Video**
4. **Monitor Progress**: Watch real-time progress bar
5. **View Result**: Video plays automatically when done

### 2. Image-to-Video

1. **Upload Start Image**: Click "Start Image" box
2. **Upload End Image** (optional): Click "End Image" box
3. **Enter Prompt**: Describe the motion/animation
4. **Generate**: Click button to start

### 3. Audio-Guided Generation

1. **Upload Audio**: Click "Audio Guide" box
2. **Enter Prompt**: Describe the visual content
3. **Generate**: Video will sync with audio

### 4. Advanced Settings

Click "Advanced Settings" to reveal:

- **Attention Mode**: Choose flash/sage/sdpa (auto is usually best)
- **Sliding Window**: For long videos (481+ frames)

## File Structure

```
ltx2_server/static/
├── index.html    # Main HTML structure
├── style.css     # Modern dark theme styles
└── app.js        # Frontend application logic
```

## Technical Details

### Frontend Stack
- **Vanilla JavaScript**: No framework dependencies
- **Modern CSS**: CSS Grid, Flexbox, Custom Properties
- **Fetch API**: Async communication with backend
- **Polling**: 1-second interval for task updates

### API Integration
- **FormData**: Handles file uploads
- **Polling Pattern**: Checks task status every second
- **Error Handling**: Graceful error messages via toasts
- **State Management**: Local task history tracking

### Responsive Breakpoints
- **Desktop**: Two-panel layout (controls + results)
- **Tablet**: Stacked single column
- **Mobile**: Optimized touch targets

## Browser Compatibility

Tested on:
- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Mobile browsers (iOS/Android)

## Customization

### Theme Colors

Edit CSS variables in `style.css`:

```css
:root {
    --primary: #6366f1;        /* Main accent color */
    --primary-hover: #4f46e5;  /* Hover state */
    --success: #10b981;        /* Success/completed */
    --warning: #f59e0b;        /* Processing/warning */
    --error: #ef4444;          /* Error/failed */
    
    --bg-primary: #0f172a;     /* Main background */
    --bg-secondary: #1e293b;   /* Card backgrounds */
    --bg-tertiary: #334155;    /* Input backgrounds */
}
```

### Default Values

Edit form defaults in `index.html`:

```html
<input id="width" value="768" ...>
<input id="height" value="512" ...>
<input id="num_frames" value="121" ...>
<input id="fps" value="24" ...>
```

### Polling Interval

Edit in `app.js`:

```javascript
state.pollInterval = setInterval(async () => {
    await pollTaskStatus(taskId);
}, 1000); // Change from 1000ms to desired interval
```

## Troubleshooting

### UI Not Loading
- Check server is running: `http://localhost:8000`
- Verify static files exist in `ltx2_server/static/`
- Check browser console for errors (F12)

### File Upload Not Working
- Verify file type matches accept attribute
- Check file size limits (if any)
- Ensure form uses `multipart/form-data`

### Progress Not Updating
- Check network tab for polling requests
- Verify task ID is correct
- Check server logs for task processing

### Video Not Playing
- Verify video format is MP4 with H.264 codec
- Check browser supports the codec
- Try downloading and playing locally

## Future Enhancements

Potential improvements:
- [ ] WebSocket for real-time progress (instead of polling)
- [ ] Batch generation queue
- [ ] Video comparison slider
- [ ] Prompt templates/presets
- [ ] Generation history persistence
- [ ] User authentication
- [ ] Video gallery with search
- [ ] Parameter comparison view
- [ ] Export/import settings
- [ ] Dark/light theme toggle

## Screenshots

The UI features:
- **Header**: Logo, title, server status indicator
- **Left Panel**: All generation controls organized by section
- **Right Panel**: Current task progress + video player
- **Bottom**: Task history list with previous generations
- **Toast Notifications**: Success/error messages in corner

## Support

For issues or questions:
1. Check browser console (F12) for errors
2. Verify server logs for backend issues
3. Ensure all dependencies are installed
4. Check API documentation at `/docs`
