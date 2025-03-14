
import React, { useState } from 'react';
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import AnimatedContainer from './AnimatedContainer';
import { usePrediction } from "../PredictionContext";

interface ResultCardProps {
  category: string;
  relatedImages: string[];
  confidence?: number;
}

const ResultCard: React.FC<ResultCardProps> = ({ 
  category,
  relatedImages,
  confidence 
}) => {
  const [loadedImages, setLoadedImages] = useState<Record<number, boolean>>({});
  const { prediction, gradcamImage, igImage, superImage } = usePrediction();

  const handleImageLoad = (index: number) => {
    setLoadedImages(prev => ({
      ...prev,
      [index]: true
    }));
  };

  return (
    <AnimatedContainer animation="slide-up" delay={200}>
      <Card className="overflow-hidden">
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <Badge className="px-3 py-1 bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
              {prediction}
            </Badge>
            {confidence !== undefined && (
              <span className="text-xs text-muted-foreground">
                Confidence: {(confidence * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </CardHeader>
        
        <Separator />
        
        <CardContent className="p-3">
          <div className="grid grid-cols-12 gap-3">
            {relatedImages.length > 0 && (
              <div className="col-span-12 aspect-video relative overflow-hidden rounded-md bg-muted shadow-subtle">
                <div 
                  className={`absolute inset-0 bg-muted animate-pulse-subtle ${loadedImages[0] ? 'hidden' : 'block'}`}
                />
                <img
                  // src={relatedImages[0]}
                  src={`data:image/jpeg;base64,${superImage}`}
                  alt={`Primary ${category}`}
                  className={`w-full h-full object-cover transition-all duration-700 ease-in-out ${loadedImages[0] ? 'opacity-100' : 'opacity-0'}`}
                  onLoad={() => handleImageLoad(0)}
                />
              </div>
            )}
            
            {relatedImages.length > 1 && (
              <div className="col-span-12 md:col-span-6 aspect-square relative overflow-hidden rounded-md bg-muted shadow-subtle">
                <div 
                  className={`absolute inset-0 bg-muted animate-pulse-subtle ${loadedImages[1] ? 'hidden' : 'block'}`}
                />
                <img
                  // src={relatedImages[1]}
                  src={`data:image/jpeg;base64,${gradcamImage}`}
                  alt={`Secondary ${category}`}
                  className={`w-full h-full object-cover transition-all duration-700 ease-in-out ${loadedImages[1] ? 'opacity-100' : 'opacity-0'}`}
                  onLoad={() => handleImageLoad(1)}
                />
              </div>
            )}

            {relatedImages.length > 2 && (
              <div className="col-span-12 md:col-span-6 aspect-square relative overflow-hidden rounded-md bg-muted shadow-subtle">
                <div 
                  className={`absolute inset-0 bg-muted animate-pulse-subtle ${loadedImages[2] ? 'hidden' : 'block'}`}
                />
                <img
                  // src={relatedImages[1]}
                  src={`data:image/jpeg;base64,${igImage}`}
                  alt={`Secondary ${category}`}
                  className={`w-full h-full object-cover transition-all duration-700 ease-in-out ${loadedImages[2] ? 'opacity-100' : 'opacity-0'}`}
                  onLoad={() => handleImageLoad(2)}
                />
              </div>
            )}
          </div>
        </CardContent>
        
        <CardFooter className="pt-1 pb-3 text-xs text-muted-foreground">
          Similar images to your uploaded content
        </CardFooter>
      </Card>
    </AnimatedContainer>
  );
};

export default ResultCard;
