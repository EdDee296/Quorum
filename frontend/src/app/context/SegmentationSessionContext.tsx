import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { ProcessedImage } from '../services/segmentationApi';

interface SegmentationSessionContextValue {
  selectedImages: File[];
  setSelectedImages: React.Dispatch<React.SetStateAction<File[]>>;
  processedImages: ProcessedImage[];
  setProcessedImages: React.Dispatch<React.SetStateAction<ProcessedImage[]>>;
  modelName: string;
  setModelName: React.Dispatch<React.SetStateAction<string>>;
  clearSession: () => void;
}

const SegmentationSessionContext = createContext<SegmentationSessionContextValue | undefined>(undefined);

const STORAGE_KEY = 'segmentation_session_v1';

interface StoredSegmentationSession {
  processedImages: ProcessedImage[];
  modelName: string;
}

export function SegmentationSessionProvider({ children }: { children: ReactNode }) {
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [processedImages, setProcessedImages] = useState<ProcessedImage[]>(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return [];
      }

      const parsed = JSON.parse(raw) as StoredSegmentationSession;

      return (parsed.processedImages ?? []).map((img) => ({
        ...img,
        file: null,
        fileSize:
          typeof img.fileSize === 'number'
            ? img.fileSize
            : typeof img.file?.size === 'number'
              ? img.file.size
              : 0,
      }));
    } catch {
      return [];
    }
  });

  const [modelName, setModelName] = useState(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return 'unetpp';
      }

      const parsed = JSON.parse(raw) as StoredSegmentationSession;
      return parsed.modelName ?? 'unetpp';
    } catch {
      return 'unetpp';
    }
  });

  useEffect(() => {
    try {
      const data: StoredSegmentationSession = {
        processedImages,
        modelName,
      };
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      // ignore storage errors
    }
  }, [processedImages, modelName]);

  const clearSession = () => {
    setSelectedImages([]);
    setProcessedImages([]);
    setModelName('unetpp');
    sessionStorage.removeItem(STORAGE_KEY);
  };

  return (
    <SegmentationSessionContext.Provider
      value={{
        selectedImages,
        setSelectedImages,
        processedImages,
        setProcessedImages,
        modelName,
        setModelName,
        clearSession,
      }}
    >
      {children}
    </SegmentationSessionContext.Provider>
  );
}

export function useSegmentationSession() {
  const context = useContext(SegmentationSessionContext);

  if (!context) {
    throw new Error('useSegmentationSession must be used inside SegmentationSessionProvider');
  }

  return context;
}