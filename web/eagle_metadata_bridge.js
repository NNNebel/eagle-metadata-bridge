import { app } from "../../scripts/app.js";

function hideWidget(node, widget) {
  if (widget.type === "hidden") return;
  widget._origType = widget.type;
  widget._origComputeSize = widget.computeSize;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4]; // -4 cancels the default margin
  node.setSize([node.size[0], node.computeSize()[1]]);
  node.graph?.change();
}

function showWidget(node, widget) {
  if (widget.type !== "hidden") return;
  widget.type = widget._origType;
  widget.computeSize = widget._origComputeSize;
  node.setSize([node.size[0], node.computeSize()[1]]);
  node.graph?.change();
}

const TARGET_CLASSES = ["EagleMetadataBridge", "EagleMetadataBridgeTest"];

app.registerExtension({
  name: "EagleMetadataBridge.ConditionalWidgets",
  async nodeCreated(node) {
    if (!TARGET_CLASSES.includes(node.comfyClass)) return;

    const get = (name) => node.widgets?.find(w => w.name === name);

    const formatWidget   = get("format");
    const compressWidget = get("compress_level");
    const qualityWidget  = get("quality");

    if (!formatWidget) return;

    const update = () => {
      const fmt = formatWidget.value;
      if (compressWidget) {
        fmt === "PNG" ? showWidget(node, compressWidget) : hideWidget(node, compressWidget);
      }
      if (qualityWidget) {
        (fmt === "WebP" || fmt === "JPEG") ? showWidget(node, qualityWidget) : hideWidget(node, qualityWidget);
      }
    };

    // Wrap existing callback instead of overwriting it
    const origCallback = formatWidget.callback;
    formatWidget.callback = function (value) {
      if (origCallback) origCallback.apply(this, arguments);
      update();
    };

    // Also apply when a saved workflow is loaded
    const origOnConfigure = node.onConfigure?.bind(node);
    node.onConfigure = function (info) {
      if (origOnConfigure) origOnConfigure(info);
      requestAnimationFrame(update);
    };

    // Apply initial state after widgets are fully initialised
    requestAnimationFrame(update);
  },
});
