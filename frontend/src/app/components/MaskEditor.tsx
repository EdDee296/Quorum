import { useState, useRef, useEffect } from 'react';
import { X, Paintbrush, Eraser, Save, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { Button } from './ui/button';
import { updateMasks } from '../services/segmentationApi';

interface MaskEditorProps {
  fileName: string;
  originalImage: string;
  chromocenterMask: string;
  nucleiMask: string;
  backgroundMask: string;
  onClose: () => void;
  onSave: (masks: { chromocenter: string; nuclei: string; background: string }) => void;
}

type MaskType = 'chromocenter' | 'nuclei' | 'background';

export function MaskEditor({
  fileName,
  originalImage,
  chromocenterMask,
  nucleiMask,
  backgroundMask,
  onClose,
  onSave,
}: MaskEditorProps) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush');
  const [activeMask, setActiveMask] = useState<MaskType>('chromocenter');
  const [brushDiameter, setBrushDiameter] = useState(8);
  const [zoom, setZoom] = useState(1);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [isSaving, setIsSaving] = useState(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);

  const [masks, setMasks] = useState({
    chromocenter: chromocenterMask,
    nuclei: nucleiMask,
    background: backgroundMask,
  });

  const originalMasks = useRef({
    chromocenter: chromocenterMask,
    nuclei: nucleiMask,
    background: backgroundMask,
  });

  useEffect(() => {
    loadCanvases();
  }, [activeMask, masks, originalImage]);

  const loadCanvases = () => {
    const baseCanvas = baseCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;

    if (!baseCanvas || !overlayCanvas) {
      return;
    }

    const baseCtx = baseCanvas.getContext('2d');
    const overlayCtx = overlayCanvas.getContext('2d');

    if (!baseCtx || !overlayCtx) {
      return;
    }

    const originalImg = new Image();
    originalImg.onload = () => {
      baseCanvas.width = originalImg.width;
      baseCanvas.height = originalImg.height;
      overlayCanvas.width = originalImg.width;
      overlayCanvas.height = originalImg.height;
      setCanvasSize({ width: originalImg.width, height: originalImg.height });

      baseCtx.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
      baseCtx.imageSmoothingEnabled = false;
      baseCtx.drawImage(originalImg, 0, 0);

      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.imageSmoothingEnabled = false;
      loadMaskAsBlackOverlay(overlayCtx, overlayCanvas.width, overlayCanvas.height, masks[activeMask]);
    };

    originalImg.src = originalImage;
  };

  const loadMaskAsBlackOverlay = (
    overlayCtx: CanvasRenderingContext2D,
    width: number,
    height: number,
    maskDataUrl: string,
  ) => {
    const maskImg = new Image();
    maskImg.onload = () => {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext('2d');

      if (!tempCtx) {
        return;
      }

      tempCtx.drawImage(maskImg, 0, 0, width, height);
      const imageData = tempCtx.getImageData(0, 0, width, height);

      for (let i = 0; i < imageData.data.length; i += 4) {
        const a = imageData.data[i + 3];
        // Treat any visible pixel as mask foreground. This preserves black brush edits.
        const isMaskPixel = a > 0;

        imageData.data[i] = 0;
        imageData.data[i + 1] = 0;
        imageData.data[i + 2] = 0;
        imageData.data[i + 3] = isMaskPixel ? 220 : 0;
      }

      overlayCtx.putImageData(imageData, 0, 0);
    };

    maskImg.src = maskDataUrl;
  };

  const getPointerPosition = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) {
      return null;
    }

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    return { x, y };
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const point = getPointerPosition(e);
    if (!point) {
      return;
    }

    lastPointRef.current = point;
    setIsDrawing(true);
    drawStroke(point, point);
  };

  const stopDrawing = () => {
    if (!isDrawing) {
      return;
    }

    lastPointRef.current = null;
    setIsDrawing(false);
    saveMaskData();
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) {
      return;
    }

    const point = getPointerPosition(e);
    if (!point) {
      return;
    }

    const lastPoint = lastPointRef.current ?? point;
    drawStroke(lastPoint, point);
    lastPointRef.current = point;
  };

  const drawStroke = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = brushDiameter;

    if (tool === 'brush') {
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = '#000000';
      ctx.strokeStyle = '#000000';
      ctx.globalAlpha = 1;
    } else {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0,0,0,1)';
      ctx.strokeStyle = 'rgba(0,0,0,1)';
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    // Ensure single-click produces a visible dot.
    ctx.beginPath();
    ctx.arc(to.x, to.y, brushDiameter / 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
  };

  const saveMaskData = () => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    // Store masks as strict binary alpha so erasing is deterministic.
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const alpha = imageData.data[i + 3] > 127 ? 255 : 0;
      imageData.data[i] = 0;
      imageData.data[i + 1] = 0;
      imageData.data[i + 2] = 0;
      imageData.data[i + 3] = alpha;
    }
    ctx.putImageData(imageData, 0, 0);

    const maskData = canvas.toDataURL('image/png');
    setMasks((prev) => ({
      ...prev,
      [activeMask]: maskData,
    }));
  };

  const handleReset = () => {
    setMasks(originalMasks.current);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await updateMasks(fileName, masks);
      onSave(masks);
      onClose();
    } catch (error) {
      console.error('Failed to save masks:', error);
      alert('Failed to save masks. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-auto m-4">
        <div className="p-4 border-b flex items-center justify-between" style={{ borderColor: '#A4CCD4' }}>
          <h2 style={{ color: '#304C64' }}>Edit Masks - {fileName}</h2>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded">
            <X className="h-5 w-5" style={{ color: '#304C64' }} />
          </button>
        </div>

        <div className="p-4">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div className="lg:col-span-3">
              <div className="relative w-full border rounded overflow-auto" style={{ borderColor: '#26788E', maxHeight: '500px' }}>
                <div
                  className="relative"
                  style={{
                    width: `${canvasSize.width * zoom}px`,
                    height: `${canvasSize.height * zoom}px`,
                  }}
                >
                  <canvas
                    ref={baseCanvasRef}
                    className="block"
                    style={{
                      width: `${canvasSize.width * zoom}px`,
                      height: `${canvasSize.height * zoom}px`,
                      imageRendering: 'pixelated',
                    }}
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="absolute top-0 left-0 cursor-crosshair"
                    style={{
                      width: `${canvasSize.width * zoom}px`,
                      height: `${canvasSize.height * zoom}px`,
                      imageRendering: 'pixelated',
                    }}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm mb-2" style={{ color: '#304C64' }}>Tool</label>
                <div className="flex gap-2">
                  <Button
                    variant={tool === 'brush' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setTool('brush')}
                    style={tool === 'brush' ? { backgroundColor: '#26788E' } : {}}
                  >
                    <Paintbrush className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={tool === 'eraser' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setTool('eraser')}
                    style={tool === 'eraser' ? { backgroundColor: '#26788E' } : {}}
                  >
                    <Eraser className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-xs mt-2" style={{ color: '#26788E' }}>
                  Brush adds black mask pixels, eraser removes them.
                </p>
              </div>

              <div>
                <label className="block text-sm mb-2" style={{ color: '#304C64' }}>Zoom</label>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setZoom((z) => Math.max(0.5, Number((z - 0.25).toFixed(2))))}
                  >
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                  <div className="text-sm min-w-16 text-center" style={{ color: '#304C64' }}>
                    {Math.round(zoom * 100)}%
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setZoom((z) => Math.min(8, Number((z + 0.25).toFixed(2))))}
                  >
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="8"
                  step="0.25"
                  value={zoom}
                  onChange={(e) => setZoom(Number(e.target.value))}
                  className="w-full mt-2"
                />
              </div>

              <div>
                <label className="block text-sm mb-2" style={{ color: '#304C64' }}>Active Mask</label>
                <div className="space-y-2">
                  <button
                    onClick={() => setActiveMask('chromocenter')}
                    className="w-full p-2 rounded"
                    style={{
                      backgroundColor: activeMask === 'chromocenter' ? '#304C64' : 'white',
                      color: activeMask === 'chromocenter' ? 'white' : '#304C64',
                      borderWidth: '1px',
                      borderColor: '#304C64',
                    }}
                  >
                    Chromocenter
                  </button>
                  <button
                    onClick={() => setActiveMask('nuclei')}
                    className="w-full p-2 rounded"
                    style={{
                      backgroundColor: activeMask === 'nuclei' ? '#304C64' : 'white',
                      color: activeMask === 'nuclei' ? 'white' : '#304C64',
                      borderWidth: '1px',
                      borderColor: '#304C64',
                    }}
                  >
                    Nuclei
                  </button>
                  <button
                    onClick={() => setActiveMask('background')}
                    className="w-full p-2 rounded"
                    style={{
                      backgroundColor: activeMask === 'background' ? '#304C64' : 'white',
                      color: activeMask === 'background' ? 'white' : '#304C64',
                      borderWidth: '1px',
                      borderColor: '#304C64',
                    }}
                  >
                    Background
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm mb-2" style={{ color: '#304C64' }}>
                  Brush Diameter: {brushDiameter}px
                </label>
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={brushDiameter}
                  onChange={(e) => setBrushDiameter(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="space-y-2 pt-4">
                <Button onClick={handleReset} variant="outline" className="w-full">
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset
                </Button>
                <Button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="w-full"
                  style={{ backgroundColor: '#26788E', color: 'white' }}
                >
                  <Save className="h-4 w-4 mr-2" />
                  {isSaving ? 'Saving...' : 'Save Changes'}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
