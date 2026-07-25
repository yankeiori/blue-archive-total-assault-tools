// SortableJS CDN ロード後にカードコンテナを初期化
(function () {
  const SORTABLE_CDN =
    "https://cdn.jsdelivr.net/npm/sortablejs@1.15.6/Sortable.min.js";

  function getOrderFromContainer(container) {
    return Array.from(container.children).map((el) => {
      // Dash はパターンマッチ ID を JSON 文字列で DOM id に設定する
      const parsed = JSON.parse(el.id);
      return parsed.index;
    });
  }

  function syncOrder(container) {
    const order = getOrderFromContainer(container);
    // Dash の set_props API で確実にコールバックを発火させる
    if (window.dash_clientside) {
      window.dash_clientside.set_props("drag-order", {
        data: JSON.stringify(order),
      });
    }
  }

  function initSortable() {
    const observer = new MutationObserver(() => {
      const container = document.getElementById("cards-container");
      if (container && !container._sortableInitialized) {
        new Sortable(container, {
          animation: 150,
          handle: ".drag-handle",
          ghostClass: "sortable-ghost",
          onEnd: function () {
            syncOrder(container);
          },
        });
        container._sortableInitialized = true;
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  const script = document.createElement("script");
  script.src = SORTABLE_CDN;
  script.onload = initSortable;
  document.head.appendChild(script);
})();
