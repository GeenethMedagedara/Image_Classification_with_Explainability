
import React, { useState, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { UploadCloud, Image as ImageIcon, X } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import AnimatedContainer from './AnimatedContainer';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  isLoading: boolean;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelect, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    // Check if file is an image
    if (!file.type.match('image.*')) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (JPEG, PNG, etc.)",
        variant: "destructive"
      });
      return;
    }
    
    // Check file size (limit to 5MB)
    if (file.size > 5 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload an image smaller than 5MB",
        variant: "destructive"
      });
      return;
    }
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    
    // Pass file to parent component
    onImageSelect(file);
  };

  const resetImage = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <AnimatedContainer animation="slide-up">
      <Card className={`relative overflow-hidden transition-all duration-300 ${dragActive ? 'ring-2 ring-primary ring-offset-2' : ''} ${isLoading ? 'opacity-80' : ''}`}>
        <CardContent className="p-0">
          {!previewUrl ? (
            <div
              className="flex flex-col items-center justify-center p-8 min-h-[300px] text-center"
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="w-16 h-16 mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <UploadCloud className="w-8 h-8 text-primary animate-pulse-subtle" />
              </div>
              <h3 className="text-lg font-medium mb-2">Upload an image</h3>
              <p className="text-muted-foreground mb-4 max-w-md">
                Drag and drop an image here, or click to browse
              </p>
              <Button 
                onClick={triggerFileInput}
                className="relative overflow-hidden transition-all"
                disabled={isLoading}
              >
                Browse files
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={handleChange}
                accept="image/*"
                disabled={isLoading}
              />
            </div>
          ) : (
            <div className="relative aspect-video w-full">
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="w-full h-full object-cover animate-scale-in"
              />
              <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/50">
                <Button 
                  variant="outline"
                  size="icon"
                  className="bg-background/80 backdrop-blur-sm text-foreground"
                  onClick={resetImage}
                  disabled={isLoading}
                >
                  <X className="w-5 h-5" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
        
        {isLoading && (
          <div className="absolute inset-0 bg-background/50 backdrop-blur-sm flex items-center justify-center">
            <div className="glass px-4 py-2 rounded-full font-medium text-sm animate-pulse-subtle">
              Analyzing image...
            </div>
          </div>
        )}
      </Card>
    </AnimatedContainer>
  );
};

export default ImageUploader;
