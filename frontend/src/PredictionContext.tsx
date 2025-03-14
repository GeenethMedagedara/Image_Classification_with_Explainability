import React, { createContext, useState, useContext, ReactNode } from "react";

interface PredictionContextType {
  prediction: string | null;
  gradcamImage: string | null;
  igImage: string | null;
  superImage: string | null;
  setPrediction: (value: string | null) => void;
  setGradcamImage: (value: string | null) => void;
  setIgImage: (value: string | null) => void;
  setSuperImage: (value: string | null) => void;
}

const PredictionContext = createContext<PredictionContextType | undefined>(undefined);

export const PredictionProvider = ({ children }: { children: ReactNode }) => {
  const [prediction, setPrediction] = useState<string | null>(null);
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);
  const [igImage, setIgImage] = useState<string | null>(null);
  const [superImage, setSuperImage] = useState<string | null>(null);

  return (
    <PredictionContext.Provider value={{ prediction, gradcamImage, igImage, superImage, setPrediction, setGradcamImage, setIgImage, setSuperImage }}>
      {children}
    </PredictionContext.Provider>
  );
};

export const usePrediction = () => {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error("usePrediction must be used within a PredictionProvider");
  }
  return context;
};
