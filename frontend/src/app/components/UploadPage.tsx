import { useState } from 'react';
import { useNavigate } from 'react-router';
import { ImageUploader } from './ImageUploader';
import { SegmentationViewer } from './SegmentationViewer';
import { MaskEditor } from './MaskEditor';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from './ui/alert-dialog';
import { processBatch, ProcessedImage, downloadAllMasks } from '../services/segmentationApi';
import { useSegmentationSession } from '../context/SegmentationSessionContext';

export function UploadPage() {
  const {
    selectedImages,
    setSelectedImages,
    processedImages,
    setProcessedImages,
    modelName,
    setModelName,
    clearSession,
  } = useSegmentationSession();

  const [isProcessing, setIsProcessing] = useState(false);
  const [processedCount, setProcessedCount] = useState(0);
  const [editingImage, setEditingImage] = useState<ProcessedImage | null>(null);
  const navigate = useNavigate();

  const displayedProcessedCount = isProcessing ? processedCount : processedImages.length;
  const totalImagesCount =
    selectedImages.length > 0 ? selectedImages.length : processedImages.length;

  const handleImagesSelected = async (files: File[]) => {
    setSelectedImages(files);
    setProcessedCount(0);
    setProcessedImages([]);
    setIsProcessing(true);

    try {
      const results = await processBatch(files, modelName, (current) => {
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
    navigate('/results');
  };

  const handleDownloadAllTiffs = async () => {
    try {
      await downloadAllMasks(processedImages);
    } catch (error: unknown) {
      console.error('Error downloading all TIFFs:', error);
      const message = error instanceof Error
        ? error.message
        : 'Failed to download all TIFFs. Please try again.';
      alert(message);
    }
  };

  const handleStartNewSession = () => {
    clearSession();
    setProcessedCount(0);
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
    <main className="max-w-5xl mx-auto px-4 py-6 sm:px-5 lg:px-6">
      {selectedImages.length === 0 && processedImages.length === 0 ? (
        <div className="max-w-4xl mx-auto space-y-6">
          <section
            className="rounded-2xl p-5 sm:p-6 shadow-sm"
            style={{
              backgroundColor: 'white',
              border: '1px solid #A4CCD4',
            }}
          >
            <div className="flex flex-col gap-5">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-sm font-medium tracking-wide uppercase" style={{ color: '#26788E' }}>
                    Start a new analysis
                  </p>
                  <h2 className="mt-1 text-2xl font-semibold" style={{ color: '#304C64' }}>
                    Upload microscopy images
                  </h2>
                  <p className="mt-2 text-sm sm:text-base max-w-2xl" style={{ color: '#5C7285' }}>
                    Choose a segmentation model, then upload one or more images to generate masks
                    for chromocenters, nuclei, and background.
                  </p>
                </div>

                <div className="sm:min-w-[220px]">
                  <label
                    className="block text-sm font-medium mb-2"
                    style={{ color: '#304C64' }}
                  >
                    Segmentation Model
                  </label>
                  <select
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="w-full rounded-xl px-4 py-3 outline-none"
                    style={{
                      border: '1px solid #26788E',
                      color: '#304C64',
                      backgroundColor: 'white',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
                    }}
                  >
                    <option value="unetpp">U-Net++ </option>
                    <option value="cellpose">Cellpose</option>
                  </select>
                </div>
              </div>

              <ImageUploader onImagesSelected={handleImagesSelected} />
            </div>
          </section>

          <section
            className="rounded-2xl p-6 sm:p-8 shadow-sm"
            style={{
              backgroundColor: '#F8FBFC',
              border: '1px solid #A4CCD4',
            }}
          >
            <div className="grid gap-6 md:grid-cols-[1.2fr_1fr]">
              <div>
                <p className="text-sm font-medium tracking-wide uppercase" style={{ color: '#26788E' }}>
                  About this tool
                </p>
                <h3 className="mt-2 text-xl font-semibold" style={{ color: '#304C64' }}>
                  Fast sub-cellular instance segmentation
                </h3>
                <p className="mt-3 text-sm sm:text-base leading-7" style={{ color: '#5C7285' }}>
                  This tool performs segmentation on microscopy images to help identify and review
                  sub-cellular structures. Upload one or more images to begin the analysis and
                  inspect the generated masks.
                </p>
              </div>

              <div
                className="rounded-2xl p-5"
                style={{
                  backgroundColor: '#A4CCD4',
                  border: '1px solid #26788E',
                }}
              >
                <h4 className="text-base font-semibold mb-3" style={{ color: '#304C64' }}>
                  Detected structures
                </h4>
                <ul className="space-y-3 text-sm sm:text-base" style={{ color: '#304C64' }}>
                  <li>
                    <strong>Chromocenter:</strong> Dense chromatin regions
                  </li>
                  <li>
                    <strong>Nuclei:</strong> Nuclear regions
                  </li>
                  <li>
                    <strong>Background:</strong> Non-nuclear areas
                  </li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      ) : (
        <div className="space-y-6">
          <div
            className="rounded-2xl p-5 sm:p-6 shadow-sm"
            style={{
              backgroundColor: 'white',
              border: '1px solid #A4CCD4',
            }}
          >
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-sm font-medium tracking-wide uppercase" style={{ color: '#26788E' }}>
                  Session overview
                </p>
                <h2 className="mt-1 text-2xl font-semibold" style={{ color: '#304C64' }}>
                  Processing Results
                </h2>
                <p className="mt-2 text-sm" style={{ color: '#5C7285' }}>
                  {displayedProcessedCount} of {totalImagesCount} images processed
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                {displayedProcessedCount === totalImagesCount && totalImagesCount > 0 && !isProcessing && (
                  <>
                    <button
                      onClick={handleDownloadAllTiffs}
                      className="px-5 py-3 text-sm font-medium rounded-xl"
                      style={{
                        backgroundColor: '#26788E',
                        color: 'white',
                        boxShadow: '0 6px 16px rgba(38, 120, 142, 0.18)',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#304C64'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#26788E'}
                    >
                      Download All TIFFs
                    </button>

                    <button
                      onClick={handleViewResults}
                      className="px-5 py-3 text-sm font-medium rounded-xl"
                      style={{
                        backgroundColor: '#26788E',
                        color: 'white',
                        boxShadow: '0 6px 16px rgba(38, 120, 142, 0.18)',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#304C64'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#26788E'}
                    >
                      View Results Summary
                    </button>
                  </>
                )}

                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <button
                      className="px-5 py-3 text-sm font-medium rounded-xl"
                      style={{
                        backgroundColor: 'white',
                        border: '1px solid #26788E',
                        color: '#304C64',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#F0F7F9'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
                    >
                      Upload New Images
                    </button>
                  </AlertDialogTrigger>

                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Start New Upload Session?</AlertDialogTitle>
                      <AlertDialogDescription>
                        Starting a new upload session will clear your current results. Please make
                        sure you have downloaded any files you need before continuing.
                      </AlertDialogDescription>
                    </AlertDialogHeader>

                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={handleStartNewSession}
                        style={{
                          backgroundColor: '#26788E',
                          color: 'white',
                          boxShadow: '0 6px 16px rgba(38, 120, 142, 0.18)',
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#304C64')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#26788E')}
                      >
                        Continue
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          </div>

          {isProcessing && (
            <div
              className="rounded-2xl py-14 px-6 text-center shadow-sm"
              style={{
                backgroundColor: 'white',
                border: '1px solid #A4CCD4',
              }}
            >
              <div
                className="inline-block animate-spin rounded-full h-14 w-14 border-4 border-solid border-current border-r-transparent"
                style={{ color: '#26788E' }}
              ></div>
              <h3 className="mt-5 text-xl font-semibold" style={{ color: '#304C64' }}>
                Processing images
              </h3>
              <p className="mt-2 text-sm sm:text-base" style={{ color: '#5C7285' }}>
                {displayedProcessedCount} / {totalImagesCount} completed
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