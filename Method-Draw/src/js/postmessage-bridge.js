/**
 * PostMessage Bridge for Method Draw
 *
 * Enables the parent window (our React app) to communicate with the
 * Method Draw editor embedded in an iframe via window.postMessage.
 *
 * Protocol:
 *   Parent → Editor:
 *     { type: "LOAD_SVG",    svg: "<svg>...</svg>" }
 *     { type: "REQUEST_SVG" }
 *
 *   Editor → Parent:
 *     { type: "CURRENT_SVG", svg: "<svg>...</svg>" }
 *     { type: "BRIDGE_READY" }
 */

(function () {
  "use strict";

  // Wait until svgCanvas is available (start.js runs before us or we
  // retry on a short interval).
  function waitForCanvas(cb) {
    if (typeof svgCanvas !== "undefined" && svgCanvas && svgCanvas.setSvgString) {
      cb();
    } else {
      setTimeout(function () { waitForCanvas(cb); }, 100);
    }
  }

  waitForCanvas(function () {
    // Tell parent we are ready
    if (window.parent && window.parent !== window) {
      window.parent.postMessage({ type: "BRIDGE_READY" }, "*");
    }

    window.addEventListener("message", function (event) {
      // Basic safety — only handle objects with a type field
      if (!event.data || typeof event.data.type !== "string") return;

      switch (event.data.type) {
        case "LOAD_SVG":
          if (typeof event.data.svg === "string" && event.data.svg.length > 0) {
            svgCanvas.setSvgString(event.data.svg);
          }
          break;

        case "REQUEST_SVG":
          var svgStr = svgCanvas.getSvgString();
          if (window.parent && window.parent !== window) {
            window.parent.postMessage({ type: "CURRENT_SVG", svg: svgStr }, "*");
          }
          break;

        default:
          // Unknown message — ignore
          break;
      }
    });
  });
})();
