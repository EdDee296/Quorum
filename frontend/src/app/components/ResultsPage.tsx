import { useLocation, useNavigate } from 'react-router';
import { ArrowLeft, Edit } from 'lucide-react';
import { useState } from 'react';
import { MaskEditor } from './MaskEditor';
import { ProcessedImage } from '../services/segmentationApi';

export function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [processedImages, setProcessedImages] = useState<ProcessedImage[]>(
    (location.state?.processedImages || []) as ProcessedImage[]
  );
  const [editingImage, setEditingImage] = useState<ProcessedImage | null>(null);

  const handleEditMask = (image: ProcessedImage) => {
    setEditingImage(image);
  };

  const handleSaveMasks = (
    fileName: string,
    masks: { chromocenter: string; nuclei: string; background: string }
  ) => {
    // Update the processed images with new masks
    setProcessedImages(prev =>
      prev.map(img =>
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
          : img
      )
    );
  };

  if (processedImages.length === 0) {
    return (
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="text-center py-12">
          <h2 style={{ color: '#304C64' }}>No results to display</h2>
          <p className="mt-2 text-sm" style={{ color: '#26788E' }}>
            Please upload and process images first.
          </p>
          <button
            onClick={() => navigate('/')}
            className="mt-4 px-4 py-2 text-sm rounded-md"
            style={{ 
              backgroundColor: '#26788E',
              color: 'white'
            }}
          >
            Go to Upload
          </button>
        </div>
      </main>
    );
  }

  return (
    <>
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-6">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 px-3 py-2 text-sm rounded-md mb-4"
            style={{ 
              backgroundColor: 'white', 
              borderColor: '#26788E', 
              borderWidth: '1px',
              color: '#304C64'
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#A4CCD4'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Upload
          </button>

          <div className="flex items-center justify-between">
            <div>
              <h2 style={{ color: '#304C64' }}>Segmentation Results</h2>
              <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                {processedImages.length} image{processedImages.length !== 1 ? 's' : ''} processed
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {processedImages.map((processedImg, index) => (
            <div
              key={index}
              className="rounded-lg p-4 flex items-center gap-4"
              style={{ backgroundColor: 'white', borderColor: '#A4CCD4', borderWidth: '1px' }}
            >
              <div className="flex-shrink-0">
                <img
                  src={processedImg.result.imageUrl}
                  alt={processedImg.fileName}
                  className="w-32 h-32 object-cover rounded"
                  style={{ borderColor: '#26788E', borderWidth: '1px' }}
                />
              </div>
              <div className="flex-grow">
                <h3 className="mb-1" style={{ color: '#304C64' }}>{processedImg.fileName}</h3>
                <p className="text-sm" style={{ color: '#26788E' }}>
                  {(processedImg.file.size / 1024).toFixed(2)} KB
                </p>
              </div>
              <button
                onClick={() => handleEditMask(processedImg)}
                className="px-4 py-2 text-sm rounded-md flex items-center gap-2"
                style={{ 
                  backgroundColor: '#26788E',
                  color: 'white'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#304C64'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#26788E'}
              >
                <Edit className="h-4 w-4" />
                Edit Masks
              </button>
            </div>
          ))}
        </div>
      </main>

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
    </>
  );
}