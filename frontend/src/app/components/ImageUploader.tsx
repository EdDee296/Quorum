import { Upload, Image as ImageIcon } from 'lucide-react';
import { Button } from './ui/button';

interface ImageUploaderProps {
  onImagesSelected: (files: File[]) => void;
}

export function ImageUploader({ onImagesSelected }: ImageUploaderProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onImagesSelected(Array.from(files));
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onImagesSelected(Array.from(files));
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      className="border-2 border-dashed rounded-2xl px-6 py-10 text-center transition-colors shadow-sm"
      style={{
        borderColor: '#A4CCD4',
        backgroundColor: '#FCFEFF',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#26788E';
        e.currentTarget.style.backgroundColor = '#F7FBFC';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#A4CCD4';
        e.currentTarget.style.backgroundColor = '#FCFEFF';
      }}
    >
      <div
        className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full"
        style={{ backgroundColor: '#EAF5F7' }}
      >
        <Upload className="h-10 w-10" style={{ color: '#26788E' }} />
      </div>

      <h3 className="text-2xl font-semibold mb-3" style={{ color: '#304C64' }}>
        Drop your images here
      </h3>

      <p className="text-sm sm:text-base mb-2" style={{ color: '#5C7285' }}>
        Drag and drop microscopy images into this area
      </p>

      <p className="text-sm mb-6" style={{ color: '#26788E' }}>
        or choose files manually from your device
      </p>

      <input
        type="file"
        id="file-upload"
        multiple
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      <div className="flex flex-col items-center gap-4">
        <Button
          asChild
          className="rounded-xl px-6 py-3 text-sm font-medium"
          style={{
            backgroundColor: '#26788E',
            color: 'white',
            boxShadow: '0 6px 16px rgba(38, 120, 142, 0.18)',
          }}
        >
          <label htmlFor="file-upload" className="cursor-pointer inline-flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Select Images
          </label>
        </Button>

        <p className="text-xs sm:text-sm" style={{ color: '#5C7285' }}>
          Supports multiple image uploads in one session
        </p>
      </div>
    </div>
  );
}