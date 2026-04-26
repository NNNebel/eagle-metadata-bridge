import { app } from "../../scripts/app.js";

// Hide a widget (collapse height to 0, remove from layout)
function hideWidget(node, widget) {
  if (widget.type === "hidden") return;
  widget._origType = widget.type;
  widget._origComputeSize = widget.computeSize;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4]; // -4 cancels the default margin
  node.setSize([node.size[0], node.computeSize()[1]]);
}

// Restore a hidden widget
function showWidget(node, widget) {
  if (widget.type !== "hidden") return;
  widget.type = widget._origType;
  widget.computeSize = widget._origComputeSize;
  node.setSize([node.size[0], node.computeSize()[1]]);
}

app.registerExtension({
  name: "EagleMetadataBridge.ConditionalWidgets",
  async nodeCreated(node) {
    const targetClasses = ["EagleMetadataBridge", "EagleMetadataBridgeTest"];
    if (!targetClasses.includes(node.comfyClass)) return;

    const get = (name) => node.widgets?.find(w => w.name === name);

    const formatWidget   = get("format");
    const compressWidget = get("compress_level");
    const qualityWidget  = get("quality");

    if (!formatWidget) return;

    const update = () => {
      const fmt = formatWidget.value;
      const isPng  = fmt === "PNG";
      const isWebp = fmt === "WebP";
      const isJpeg = fmt === "JPEG";

      if (compressWidget) isPng  ? showWidget(node, compressWidget) : hideWidget(node, compressWidget);
      if (qualityWidget)  (isWebp || isJpeg) ? showWidget(node, qualityWidget) : hideWidget(node, qualityWidget);
    };

    formatWidget.callback = (value) => { update(); };
    update();
  },
});
