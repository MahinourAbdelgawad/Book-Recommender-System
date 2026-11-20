document.querySelectorAll(".cover img").forEach((image) => {
  const showPlaceholder = () => image.remove();
  image.addEventListener("error", showPlaceholder, { once: true });
  if (image.complete && image.naturalWidth === 0) showPlaceholder();
});
