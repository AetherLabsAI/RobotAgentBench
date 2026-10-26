# RobotAgentBench — project website

Static site for **RobotAgentBench: Towards Open-world Robot Autonomy** (no build step).

Preview locally (the script supports HTTP range requests, so videos can be seeked):

    python3 serve.py 8000   # then open http://localhost:8000

## Layout
- `index.html` — page structure and copy
- `styles.css`, `branding.css`, `task-preview.css` — styles
- `app.js` — demo player, leaderboard data and tables, lightbox, lazy video loading
- `task-preview.js`, `assets/task-previews.js` — task preview strip and its data
- `assets/fig/` — paper figures (`*_1600.webp` are the on-page copies; the full-size file opens on click)
- `assets/video/` — demo, looping clips, task previews (`task-previews/thumbs/` are the small card clips)
- `assets/img/` — project and institution logos, WeChat group QR code

## Updating
- Leaderboard: edit the `T1`, `T2`, `T3` arrays in `app.js`; ranks, medals and highlights are computed.
- Links still marked `data-todo` in `index.html`: Paper (arXiv), Code, Dataset, Feishu, Discord.
- WeChat group QR codes expire after 7 days: replace `assets/img/wechat-qr.png` and bump its `?v=` in `index.html`.
