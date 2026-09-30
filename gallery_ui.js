/* Copy original image bytes. Animated GIFs are never converted to still PNGs. */
(() => {
  'use strict';
  const assets = JSON.parse(document.getElementById('clipboard-assets').textContent);
  const blobCache = new Map();
  const gifLoads = new Map();
  const gifData = new Map();
  window.registerGalleryGif = (name, base64) => { gifData.set(name, base64); };

  function decodeBlob(base64, mime) {
    const binary = atob(base64);
    return new Blob([Uint8Array.from(binary, ch => ch.charCodeAt(0))], {type: mime});
  }

  function originalImage(name) {
    const asset = assets[name];
    if (!asset) throw new Error('Image is missing from this gallery.');
    if (blobCache.has(name)) return blobCache.get(name);
    if (asset.base64) {
      const blob = decodeBlob(asset.base64, asset.mime);
      blobCache.set(name, blob);
      return blob;
    }
    if (!gifLoads.has(name)) {
      gifLoads.set(name, new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = asset.script;
        script.onload = () => {
          try {
            if (!gifData.has(name)) throw new Error('GIF clipboard data missing.');
            const blob = decodeBlob(gifData.get(name), asset.mime);
            blobCache.set(name, blob);
            gifData.delete(name);
            resolve(blob);
          } catch (error) { reject(error); }
          finally { script.remove(); }
        };
        script.onerror = () => {
          script.remove();
          reject(new Error('GIF clipboard data could not be loaded.'));
        };
        document.head.appendChild(script);
      }));
    }
    return gifLoads.get(name);
  }

  const canWrite = () => navigator.clipboard &&
    typeof navigator.clipboard.write === 'function' && typeof ClipboardItem !== 'undefined';
  const explicitlyUnsupported = mime => typeof ClipboardItem !== 'undefined' &&
    typeof ClipboardItem.supports === 'function' && !ClipboardItem.supports(mime);

  document.querySelectorAll('[data-copy-image]').forEach(button => {
    const name = button.dataset.copyImage;
    const asset = assets[name];
    const isGif = asset.mime === 'image/gif';
    const restingLabel = button.textContent;
    const originalAriaLabel = button.getAttribute('aria-label');
    const status = document.getElementById(button.getAttribute('aria-describedby'));
    let downloadMode = false;
    let resetTimer;

    function offerGifDownload(message) {
      downloadMode = true;
      button.textContent = 'Download GIF';
      button.setAttribute('aria-label', originalAriaLabel.replace('Copy GIF', 'Download GIF'));
      status.textContent = message + ' Download the GIF, then insert it into Google Slides as an image.';
      status.dataset.state = 'notice';
    }
    if (isGif && (!canWrite() || explicitlyUnsupported(asset.mime))) {
      offerGifDownload('This browser cannot copy animated GIF files.');
    }

    button.addEventListener('click', async () => {
      clearTimeout(resetTimer);
      if (downloadMode) {
        const link = document.createElement('a');
        link.href = 'figures/' + name;
        link.download = name;
        document.body.appendChild(link);
        link.click();
        link.remove();
        status.textContent = 'GIF download requested. In Google Slides, use Insert → Image → Upload from computer.';
        status.dataset.state = 'notice';
        return;
      }
      status.textContent = '';
      button.disabled = true;
      button.textContent = 'Copying…';
      try {
        if (!canWrite()) throw new Error('Image clipboard is unavailable.');
        // A Promise-valued GIF payload lets write start during user activation,
        // while local GIF bytes load lazily without file:// fetch restrictions.
        await navigator.clipboard.write([
          new ClipboardItem({[asset.mime]: originalImage(name)})
        ]);
        button.textContent = 'Copied ✓';
        status.textContent = isGif
          ? 'Original animated GIF copied. If your slide app pastes a still image, use Download GIF and insert the file.'
          : `Copied ${asset.width} × ${asset.height} PNG. Paste into your slide.`;
        status.dataset.state = 'success';
        resetTimer = setTimeout(() => { button.textContent = restingLabel; }, 2500);
      } catch (error) {
        if (isGif) {
          offerGifDownload('GIF copying was blocked or unsupported; no still image was substituted.');
        } else {
          button.textContent = restingLabel;
          status.textContent = 'This browser blocked image copying. Use Open PNG, then right-click and choose Copy image, or download it.';
          status.dataset.state = 'error';
        }
      } finally {
        button.disabled = false;
      }
    });
  });
})();
