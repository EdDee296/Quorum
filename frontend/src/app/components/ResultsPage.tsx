import { useNavigate } from 'react-router';
import { ArrowLeft, Edit } from 'lucide-react';
import { useState } from 'react';
import { MaskEditor } from './MaskEditor';
import { ProcessedImage } from '../services/segmentationApi';
import { useSegmentationSession } from '../context/SegmentationSessionContext';

export function ResultsPage() {
  const navigate = useNavigate();
  const { processedImages, setProcessedImages } = useSegmentationSession();
  const [editingImage, setEditingImage] = useState<ProcessedImage | null>(null);
  const [showAllImages, setShowAllImages] = useState(false);

  const handleEditMask = (image: ProcessedImage) => {
    setEditingImage(image);
  };

  const handleSaveMasks = (
    fileName: string,
    masks: { chromocenter: string; nuclei: string; background: string }
  ) => {
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

  const summaries = processedImages
    .map((p) => p.result.summary)
    .filter(Boolean);

  const totalNuclei = summaries.reduce((sum, s) => sum + (s?.nuclei_count ?? 0), 0);
  const totalChromocenters = summaries.reduce((sum, s) => sum + (s?.chromocenter_count ?? 0), 0);

  const avgChromocentersPerNucleus =
    totalNuclei > 0 ? (totalChromocenters / totalNuclei).toFixed(2) : '0.00';

  const uncertaintySummaries = processedImages
    .map((p) => p.result.uncertaintySummary)
    .filter(Boolean);

  const avgConfidence = uncertaintySummaries.length > 0
    ? (
        uncertaintySummaries.reduce((sum, u) => sum + (u?.mean_confidence ?? 0), 0) /
        uncertaintySummaries.length
      ).toFixed(2)
    : '0.00';

  const avgEntropy = uncertaintySummaries.length > 0
    ? (
        uncertaintySummaries.reduce((sum, u) => sum + (u?.normalized_mean_entropy ?? 0), 0) /
        uncertaintySummaries.length
      ).toFixed(2)
    : '0.00';

  const totalNeedsReview = processedImages.reduce(
    (sum, img) => sum + (img.result.uncertaintySummary?.needs_review ? 1 : 0),
    0
  );

  const visibleProcessedImages = showAllImages ? processedImages : processedImages.slice(0, 10);

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
            Back to Results
          </button>

          <div className="flex items-center justify-between">
            <div>
              <h2 style={{ color: '#304C64' }}>Results Summary</h2>
              <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                {processedImages.length} image{processedImages.length !== 1 ? 's' : ''} processed
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div
            className="rounded-lg p-4"
            style={{ backgroundColor: 'white', borderColor: '#A4CCD4', borderWidth: '1px' }}
          >
            <p className="text-sm" style={{ color: '#26788E' }}>Images Processed</p>
            <h3 className="mt-2" style={{ color: '#304C64' }}>{processedImages.length}</h3>
          </div>

          <div
            className="rounded-lg p-4"
            style={{ backgroundColor: 'white', borderColor: '#A4CCD4', borderWidth: '1px' }}
          >
            <p className="text-sm" style={{ color: '#26788E' }}>Detected Nuclei</p>
            <h3 className="mt-2" style={{ color: '#304C64' }}>{totalNuclei}</h3>
          </div>

          <div
            className="rounded-lg p-4"
            style={{ backgroundColor: 'white', borderColor: '#A4CCD4', borderWidth: '1px' }}
          >
            <p className="text-sm" style={{ color: '#26788E' }}>Detected Chromocenters</p>
            <h3 className="mt-2" style={{ color: '#304C64' }}>{totalChromocenters}</h3>
          </div>

          <div
            className="rounded-lg p-4"
            style={{ backgroundColor: 'white', borderColor: '#A4CCD4', borderWidth: '1px' }}
          >
            <p className="text-sm" style={{ color: '#26788E' }}>Avg Chromocenters / Nucleus</p>
            <h3 className="mt-2" style={{ color: '#304C64' }}>{avgChromocentersPerNucleus}</h3>
          </div>
        </div>

        <div
          className="rounded-lg p-5 mb-8"
          style={{ backgroundColor: 'white', borderColor: '#A4CCD4', borderWidth: '1px' }}
        >
          <h3 className="mb-2" style={{ color: '#304C64' }}>Processed Images Overview</h3>
          <p className="text-sm mb-4" style={{ color: '#26788E' }}>
            Quick access to each processed image so you can review and edit masks easily.
          </p>

          <div className="space-y-4">
            {visibleProcessedImages.map((processedImg, index) => (
              <div
                key={index}
                className="rounded-lg p-4 flex items-center gap-4"
                style={{ backgroundColor: '#F8FBFC', borderColor: '#A4CCD4', borderWidth: '1px' }}
              >
                <div className="flex-shrink-0">
                  <img
                    src={processedImg.result.imageUrl}
                    alt={processedImg.fileName}
                    className="w-28 h-28 object-cover rounded"
                    style={{ borderColor: '#26788E', borderWidth: '1px' }}
                  />
                </div>

                <div className="flex-grow">
                  <h3 className="mb-1" style={{ color: '#304C64' }}>{processedImg.fileName}</h3>
                  <p className="text-sm" style={{ color: '#26788E' }}>
                    {(processedImg.fileSize / 1024).toFixed(2)} KB
                  </p>
                  <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                    Nuclei: {processedImg.result.summary?.nuclei_count ?? 0} | Chromocenters: {processedImg.result.summary?.chromocenter_count ?? 0}
                  </p>
                  {processedImg.result.uncertaintySummary && (
                    <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                      Confidence: {processedImg.result.uncertaintySummary.confidence_label} | Score: {(processedImg.result.uncertaintySummary.mean_confidence * 100).toFixed(1)}%
                    </p>
                  )}
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

          {processedImages.length > 10 && (
            <div className="mt-4">
              <button
                onClick={() => setShowAllImages((prev) => !prev)}
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
                {showAllImages ? 'Show Less' : 'Show All'}
              </button>
            </div>
          )}
        </div>

        <div
          className="rounded-lg p-5"
          style={{ backgroundColor: 'white', borderColor: '#A4CCD4', borderWidth: '1px' }}
        >
          <h3 className="mb-2" style={{ color: '#304C64' }}>Prediction Confidence Summary</h3>
          <p className="text-sm mb-4" style={{ color: '#26788E' }}>
            Confidence is estimated using multiple test-time augmentation views. Higher entropy may indicate that the prediction should be reviewed manually.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div
              className="rounded-lg p-4"
              style={{ backgroundColor: '#F8FBFC', borderColor: '#A4CCD4', borderWidth: '1px' }}
            >
              <p className="text-sm" style={{ color: '#26788E' }}>Average Confidence</p>
              <h3 className="mt-2" style={{ color: '#304C64' }}>{(Number(avgConfidence) * 100).toFixed(1)}%</h3>
            </div>

            <div
              className="rounded-lg p-4"
              style={{ backgroundColor: '#F8FBFC', borderColor: '#A4CCD4', borderWidth: '1px' }}
            >
              <p className="text-sm" style={{ color: '#26788E' }}>Average Entropy</p>
              <h3 className="mt-2" style={{ color: '#304C64' }}>{Number(avgEntropy).toFixed(3)}</h3>
            </div>

            <div
              className="rounded-lg p-4"
              style={{ backgroundColor: '#F8FBFC', borderColor: '#A4CCD4', borderWidth: '1px' }}
            >
              <p className="text-sm" style={{ color: '#26788E' }}>Images Needing Review</p>
              <h3 className="mt-2" style={{ color: '#304C64' }}>{totalNeedsReview}</h3>
            </div>
          </div>

          <div className="space-y-4">
            {processedImages.map((processedImg, index) => {
              const uncertainty = processedImg.result.uncertaintySummary;

              return (
                <div
                  key={index}
                  className="rounded-lg p-4 flex items-center justify-between gap-4"
                  style={{ backgroundColor: '#F8FBFC', borderColor: '#A4CCD4', borderWidth: '1px' }}
                >
                  <div>
                    <h4 style={{ color: '#304C64' }}>{processedImg.fileName}</h4>

                    {uncertainty ? (
                      <>
                        <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                          Confidence: {uncertainty.confidence_label}
                        </p>
                        <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                          Confidence: {(uncertainty.mean_confidence * 100).toFixed(1)}% | Entropy: {uncertainty.normalized_mean_entropy.toFixed(3)}
                        </p>
                        <p className="text-sm mt-1" style={{ color: uncertainty.confidence_label === 'Low' ? '#E2480C' : '#26788E' }}>
                          Prediction stability: {uncertainty.confidence_label}
                        </p>
                      </>
                    ) : (
                      <p className="text-sm mt-1" style={{ color: '#26788E' }}>
                        No confidence information available for this image.
                      </p>
                    )}
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
                    Review / Edit
                  </button>
                </div>
              );
            })}
          </div>
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