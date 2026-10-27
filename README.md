# adityakulshrestha.github.io

Hugo site, no theme dependency. Layouts live in `layouts/`, styles in `assets/css/main.css`.

## Preview

```
winget install Hugo.Hugo.Extended
hugo server -D
```

## Add a blog post

```
hugo new content posts/my-post.md
```

Edit `content/posts/my-post.md`, set `description` and `tags`, change `draft: true` to `false`, then push to `main`. It appears on the Blog tab and the home page automatically. Images go in `static/assets/<post>/` and are referenced as `/assets/<post>/img.png`.

## Add slideshow photos

Drop images (`.jpg`, `.png`, `.webp`) into `assets/photos/`. They are resized and shown in the home page slideshow; the file name becomes the alt text.

## Edit the home page

- Bio: `content/_index.md`
- Updates and talks (newest first): `data/home.yaml`
- Name, role, focus areas, social links, profile photo: `hugo.toml` under `[params]`

## Deploy

Push to `main`; GitHub Actions builds and publishes to Pages.
