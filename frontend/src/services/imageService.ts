
// This is a mock service that simulates a backend API for image classification
// In a real application, you would replace this with actual API calls

// Sample categories with related images
const categories = [
  {
    name: 'Nature',
    images: [
      'https://images.unsplash.com/photo-1469474968028-56623f02e42e?q=80&w=600&auto=format',
      'https://images.unsplash.com/photo-1426604966848-d7adac402bff?q=80&w=600&auto=format'
    ]
  },
  {
    name: 'Architecture',
    images: [
      'https://images.unsplash.com/photo-1616578338771-88ada8c36658?q=80&w=600&auto=format',
      'https://images.unsplash.com/photo-1470290449668-02dd93d9420a?q=80&w=600&auto=format'
    ]
  },
  {
    name: 'Food',
    images: [
      'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?q=80&w=600&auto=format',
      'https://images.unsplash.com/photo-1606787366850-de6330128bfc?q=80&w=600&auto=format'
    ]
  },
  {
    name: 'Technology',
    images: [
      'https://images.unsplash.com/photo-1498050108023-c5249f4df085?q=80&w=600&auto=format',
      'https://images.unsplash.com/photo-1520869562399-e772f042f422?q=80&w=600&auto=format'
    ]
  },
  {
    name: 'People',
    images: [
      'https://images.unsplash.com/photo-1522202176988-66273c2fd55f?q=80&w=600&auto=format',
      'https://images.unsplash.com/photo-1506277886164-e25aa3f4ef7f?q=80&w=600&auto=format'
    ]
  }
];

// Function to simulate uploading and analyzing an image
export const analyzeImage = async (file: File): Promise<{
  category: string;
  confidence: number;
  relatedImages: string[];
}> => {
  // Simulate network delay
  return new Promise((resolve) => {
    setTimeout(() => {
      // Randomly select a category (in real app this would be determined by ML model)
      const randomIndex = Math.floor(Math.random() * categories.length);
      const selectedCategory = categories[randomIndex];
      
      resolve({
        category: selectedCategory.name,
        confidence: 0.7 + (Math.random() * 0.3), // Random confidence between 70% and 100%
        relatedImages: selectedCategory.images
      });
    }, 2000); // 2 second delay to simulate processing
  });
};
