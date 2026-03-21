import { useMemo } from 'react';
import { Card } from './ui/card';
import { ProcessedImage, downloadMasks } from '../services/segmentationApi';
import { Edit } from 'lucide-react';

interface SegmentationViewerProps {
  processedImage: ProcessedImage;
  onEdit?: (image: ProcessedImage) => void;
}

export function SegmentationViewer({ processedImage, onEdit }: SegmentationViewerProps) {
  const originalImageSrc = useMemo(() => {
    if (processedImage.result.imageUrl) {
      return processedImage.result.imageUrl;
    }

    // Fallback for older state payloads.
    return URL.createObjectURL(processedImage.file);
  }, [processedImage]);

  return (
    <Card className="p-3 border-[#304C64]">
      <div className="flex items-center justify-between mb-3">
        <h3 style={{ color: '#304C64' }}>{processedImage.fileName}</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => downloadMasks(processedImage)}
            className="px-3 py-2 text-sm rounded-md flex items-center gap-2"
            style={{ backgroundColor: 'white', color: '#304C64', borderColor: '#26788E', borderWidth: '1px' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#A4CCD4';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'white';
            }}
          >
            Download TIFF
          </button>

          {onEdit && (
            <button
              onClick={() => onEdit(processedImage)}
              className="px-3 py-2 text-sm rounded-md flex items-center gap-2"
              style={{ backgroundColor: '#26788E', color: 'white' }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#304C64';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#26788E';
              }}
            >
              <Edit className="h-4 w-4" />
              Edit Masks
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Original Image */}
        <div>
          <h3 className="mb-2 text-sm" style={{ color: '#304C64' }}>Original</h3>
          <div className="relative rounded overflow-hidden" style={{ backgroundColor: '#A4CCD4' }}>
            <img
              src={originalImageSrc}
              alt="Original"
              className="w-full h-auto min-h-32 object-contain"
            />
          </div>
        </div>

        {/* Segmented Image */}
        <div>
          <h3 className="mb-2 text-sm" style={{ color: '#304C64' }}>Segmented</h3>
          <div className="relative rounded overflow-hidden" style={{ backgroundColor: '#A4CCD4' }}>
            <img
              src={originalImageSrc}
              alt="Segmented"
              className="w-full h-auto min-h-32 object-contain"
            />
            <img
              src={processedImage.result.chromocenterMask}
              alt="Chromocenter overlay"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              style={{ opacity: 0.8 }}
            />
          </div>
        </div>
      </div>

      {/* Segmentation Masks */}
      <div className="mt-4 pt-3 border-t" style={{ borderColor: '#A4CCD4' }}>
        <h4 className="text-sm mb-3" style={{ color: '#304C64' }}>Segmentation Masks:</h4>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#E2480C' }}></div>
              <span className="text-sm" style={{ color: '#304C64' }}>Chromocenter</span>
            </div>
            <img
              src={processedImage.result.chromocenterMask}
              alt="Chromocenter mask"
              className="w-full h-auto rounded min-h-32 object-contain"
              style={{ backgroundColor: '#f4f4f4' }}
            />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#26788E' }}></div>
              <span className="text-sm" style={{ color: '#304C64' }}>Nuclei</span>
            </div>
            <img
              src={processedImage.result.nucleiMask}
              alt="Nuclei mask"
              className="w-full h-auto rounded min-h-32 object-contain"
              style={{ backgroundColor: '#f4f4f4' }}
            />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#A4CCD4' }}></div>
              <span className="text-sm" style={{ color: '#304C64' }}>Background</span>
            </div>
            <img
              src={processedImage.result.backgroundMask}
              alt="Background mask"
              className="w-full h-auto rounded min-h-32 object-contain"
              style={{ backgroundColor: '#f4f4f4' }}
            />
          </div>
        </div>
      </div>
    </Card>
  );
}