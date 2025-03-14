
import React, { ReactNode } from 'react';
import { cn } from "@/lib/utils";

interface AnimatedContainerProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  animation?: 'fade-in' | 'slide-up' | 'scale-in' | 'none';
}

const AnimatedContainer: React.FC<AnimatedContainerProps> = ({
  children,
  className,
  delay = 0,
  animation = 'fade-in'
}) => {
  const getAnimationClass = () => {
    switch (animation) {
      case 'fade-in':
        return 'animate-fade-in';
      case 'slide-up':
        return 'animate-slide-up';
      case 'scale-in':
        return 'animate-scale-in';
      case 'none':
        return '';
      default:
        return 'animate-fade-in';
    }
  };

  return (
    <div 
      className={cn(getAnimationClass(), className)}
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'both' }}
    >
      {children}
    </div>
  );
};

export default AnimatedContainer;
