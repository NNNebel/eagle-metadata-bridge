import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "EagleMetadataBridge.ConditionalWidgets",
  async nodeCreated(node) {
    if (node.comfyClass !== "EagleMetadataBridge") return;

    const formatWidget = node.widgets?.find(w => w.name === "format");
    const compressWidget = node.widgets?.find(w => w.name === "compress_level");
    const qualityWidget = node.widgets?.find(w => w.name === "quality");

    if (!formatWidget || !compressWidget || !qualityWidget) return;

    const update = () => {
      const isPng = formatWidget.value === "PNG";
      compressWidget.disabled = !isPng;
      qualityWidget.disabled = isPng;
      node.setDirtyCanvas(true);
    };

    formatWidget.callback = update;
    update();
  },
});
