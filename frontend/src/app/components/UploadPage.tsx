import { useState } from 'react';
import { useNavigate } from 'react-router';
import { ImageUploader } from './ImageUploader';
import { SegmentationViewer } from './SegmentationViewer';
import { MaskEditor } from './MaskEditor';
import { processBatch, ProcessedImage } from '../services/segmentationApi';

export function UploadPage() {
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [processedImages, setProcessedImages] = useState<ProcessedImage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedCount, setProcessedCount] = useState(0);
  const [editingImage, setEditingImage] = useState<ProcessedImage | null>(null);
  const navigate = useNavigate();

  const handleImagesSelected = async (files: File[]) => {
    setSelectedImages(files);
    setProcessedCount(0);
    setProcessedImages([]);
    setIsProcessing(true);

    try {
      // Process images using the API service
      const results = await processBatch(files, (current, total) => {
        setProcessedCount(current);
      });
      
      setProcessedImages(results);
    } catch (error: unknown) {
      console.error('Error processing images:', error);
      const message = error instanceof Error
        ? error.message
        : 'Failed to process images. Please try again.';
      alert(message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleViewResults = () => {
    navigate('/results', { state: { processedImages } });
  };

  const handleSaveMasks = (
    fileName: string,
    masks: { chromocenter: string; nuclei: string; background: string },
  ) => {
    setProcessedImages((prev) =>
      prev.map((img) =>
        img.fileName === fileName
          ? {
              ...img,
              result: {
                ...img.result,
                chromocenterMask: masks.chromocenter,
                nucleiMask: masks.nuclei,
                backgroundMask: masks.background,
              },
            }
          : img,
      ),
    );
  };

  return (
    <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
      {selectedImages.length === 0 ? (
        <div className="max-w-2xl mx-auto">
          <ImageUploader onImagesSelected={handleImagesSelected} />
          <div className="mt-8 rounded-lg p-4" style={{ backgroundColor: '#A4CCD4', borderColor: '#26788E', borderWidth: '1px' }}>
            <h3 className="text-sm mb-2" style={{ color: '#304C64' }}>About this tool</h3>
            <p className="text-sm" style={{ color: '#304C64' }}>
              This tool performs instance segmentation on microscopy images to identify and
              classify sub-cellular structures. Upload one or more images to get started.
              The segmentation will identify:
            </p>
            <ul className="mt-2 text-sm list-disc list-inside space-y-1" style={{ color: '#304C64' }}>
              <li><strong>Chromocenter:</strong> Dense chromatin regions</li>
              <li><strong>Nuclei:</strong> Nuclear regions</li>
              <li><strong>Background:</strong> Non-nuclear areas</li>
            </ul>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 style={{ color: '#304C64' }}>Processing Results</h2>
              <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                {processedCount} of {selectedImages.length} images processed
              </p>
            </div>
            <div className="flex gap-3">
              {processedCount === selectedImages.length && !isProcessing && (
                <button
                  onClick={handleViewResults}
                  className="px-4 py-2 text-sm rounded-md"
                  style={{ 
                    backgroundColor: '#26788E',
                    color: 'white'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#304C64'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#26788E'}
                >
                  View Results Summary
                </button>
              )}
              <button
                onClick={() => setSelectedImages([])}
                className="px-4 py-2 text-sm rounded-md"
                style={{ 
                  backgroundColor: 'white', 
                  borderColor: '#26788E', 
                  borderWidth: '1px',
                  color: '#304C64'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#A4CCD4'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
              >
                Upload New Images
              </button>
            </div>
          </div>

          {isProcessing && (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-solid border-current border-r-transparent" 
                   style={{ color: '#26788E' }}></div>
              <p className="mt-4" style={{ color: '#304C64' }}>
                Processing images... {processedCount} / {selectedImages.length}
              </p>
            </div>
          )}

          {!isProcessing && processedImages.length > 0 && (
            <div className="space-y-6">
              {processedImages.map((processedImg, index) => (
                <div key={index}>
                  <SegmentationViewer
                    processedImage={processedImg}
                    onEdit={(img) => setEditingImage(img)}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {editingImage && (
        <MaskEditor
          fileName={editingImage.fileName}
          originalImage={editingImage.result.imageUrl}
          chromocenterMask={editingImage.result.chromocenterMask}
          nucleiMask={editingImage.result.nucleiMask}
          backgroundMask={editingImage.result.backgroundMask}
          onClose={() => setEditingImage(null)}
          onSave={(masks) => {
            handleSaveMasks(editingImage.fileName, masks);
          }}
        />
      )}
    </main>
  );
}