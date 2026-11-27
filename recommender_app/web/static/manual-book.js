{
  const form = document.querySelector("[data-suggestions-url]");
  if (form) {
    const {title, author, existing_id: existing} = form.elements;
    const list = document.querySelector("#manual-suggestions");
    let timer;
    let request;
    author.addEventListener("input", () => { existing.value = ""; });
    title.addEventListener("input", () => {
      existing.value = "";
      author.required = true;
      clearTimeout(timer);
      request?.abort();
      list.replaceChildren();
      list.hidden = true;
      if (title.value.trim().length < 2) return;
      timer = setTimeout(async () => {
        const controller = new AbortController();
        request = controller;
        const url = new URL(form.dataset.suggestionsUrl, location.href);
        url.searchParams.set("title", title.value);
        try {
          const response = await fetch(url, {signal: controller.signal});
          if (!response.ok) return;
          const {books} = await response.json();
          if (controller.signal.aborted) return;
          for (const book of books) {
            const item = document.createElement("li");
            const choice = document.createElement("button");
            choice.type = "button";
            choice.textContent = [book.title, book.author, book.year].filter(Boolean).join(" · ");
            choice.addEventListener("click", () => {
              title.value = book.title;
              author.value = book.author;
              author.required = Boolean(book.author);
              existing.value = book.id;
              list.hidden = true;
              form.querySelector('[type="submit"]').focus();
            });
            item.append(choice);
            list.append(item);
          }
          list.hidden = books.length === 0;
        } catch (error) {
          if (error.name !== "AbortError") console.warn("Book suggestions unavailable", error);
        }
      }, 200);
    });
  }
}
