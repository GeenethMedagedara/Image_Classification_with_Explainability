import React, { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { toast } from "@/components/ui/use-toast";
import ImageUploader from '@/components/ImageUploader';
import ResultCard from '@/components/ResultCard';
import AnimatedContainer from '@/components/AnimatedContainer';
import { analyzeImage } from '@/services/imageService';
import axios from 'axios';
import { set } from 'date-fns';
import { usePrediction } from '../PredictionContext';

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{
    category: string;
    confidence: number;
    relatedImages: string[];
  } | null>(null);
  // const [prediction, setPrediction] = useState(null);
  // const [gradcamImage, setGradcamImage] = useState(null);
  // const [igImage, setIgImage] = useState(null);
  // const [superImage, setSuperImage] = useState(null);
  const { setPrediction, setGradcamImage, setIgImage, setSuperImage } = usePrediction();

  const handleImageSelect = async (file: File) => {
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const data = await analyzeImage(file);
      console.log(file);
      const req = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log(req.data);
      setPrediction(req.data.predicted_class);
      setGradcamImage(req.data.gradcam_image);
      setIgImage(req.data.integrated_gradients_image);
      setSuperImage(req.data.superimposed_image);
      
      setResult(data);
      
      toast({
        title: "Analysis complete",
        description: `Your image has been classified as ${data.category}`,
      });
    } catch (error) {
      console.error('Error analyzing image:', error);
      
      toast({
        title: "Analysis failed",
        description: "There was an error analyzing your image. Please try again.",
        variant: "destructive",
      });
      
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="py-4 border-b bg-background/95 backdrop-blur-sm sticky top-0 z-10">
        <div className="container px-4 md:px-6">
          <AnimatedContainer animation="fade-in">
            <h1 className="text-center text-2xl md:text-3xl font-semibold tracking-tight">
              Intelligent Image Recognition
            </h1>
            <p className="mt-1 text-center text-muted-foreground text-sm max-w-2xl mx-auto">
              Upload any image and our machine learning model will identify its category
            </p>
          </AnimatedContainer>
        </div>
      </header>

      <main className="flex-1 py-6">
        <div className="container px-4 md:px-6 max-w-6xl">
          <div className="grid md:grid-cols-4 gap-6">
            <div className="md:col-span-1 space-y-4">
              <AnimatedContainer animation="fade-in">
                <h2 className="text-lg font-medium mb-2">Upload Image</h2>
              </AnimatedContainer>
              
              <ImageUploader 
                onImageSelect={handleImageSelect}
                isLoading={isLoading}
              />
            </div>
            
            <div className="md:col-span-3 space-y-4">
              <AnimatedContainer animation="fade-in">
                <h2 className="text-lg font-medium mb-2">Classification Results</h2>
              </AnimatedContainer>
              
              {result ? (
                <ResultCard 
                  category={result.category}
                  relatedImages={result.relatedImages}
                  confidence={result.confidence}
                />
              ) : (
                <AnimatedContainer animation="fade-in" delay={200}>
                  <Card className="min-h-[500px] flex flex-col items-center justify-center text-center p-8">
                    <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="text-muted-foreground"
                      >
                        <rect width="18" height="18" x="3" y="3" rx="2" />
                        <circle cx="9" cy="9" r="2" />
                        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-medium mb-2">No results yet</h3>
                    <p className="text-muted-foreground">
                      Upload an image to see the classification results and related images
                    </p>
                  </Card>
                </AnimatedContainer>
              )}
            </div>
          </div>
        </div>
      </main>
      
      <footer className="py-4 border-t">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <p className="text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} Image Classifier. All rights reserved.
            </p>
            <p className="text-xs text-muted-foreground">
              Powered by advanced machine learning technology
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
