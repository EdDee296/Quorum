import { Upload } from 'lucide-react';
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
      className="border-2 border-dashed rounded-lg p-12 text-center transition-colors"
      style={{ borderColor: '#A4CCD4' }}
      onMouseEnter={(e) => e.currentTarget.style.borderColor = '#26788E'}
      onMouseLeave={(e) => e.currentTarget.style.borderColor = '#A4CCD4'}
    >
      <Upload className="mx-auto h-12 w-12 mb-4" style={{ color: '#26788E' }} />
      <h3 className="mb-2" style={{ color: '#304C64' }}>Drop your images here</h3>
      <p className="text-sm mb-4" style={{ color: '#26788E' }}>or click to browse</p>
      <input
        type="file"
        id="file-upload"
        multiple
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
      <Button asChild style={{ backgroundColor: '#26788E', color: 'white' }}>
        <label htmlFor="file-upload" className="cursor-pointer">
          Select Images
        </label>
      </Button>
    </div>
  );
}