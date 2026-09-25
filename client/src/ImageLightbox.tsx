import { useEffect, useRef } from "react";

type ImageLightboxProps = {
  src: string;
  alt: string;
  onClose: () => void;
};

function ImageLightbox({ src, alt, onClose }: ImageLightboxProps) {
  const overlayRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    overlayRef.current?.focus();
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") event.preventDefault();
    };
    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
      opener?.focus();
    };
  }, [onClose]);

  return (
    <div
      ref={overlayRef}
      className="image-lightbox-overlay"
      role="dialog"
      aria-modal="true"
      aria-label={alt}
      tabIndex={-1}
      onClick={onClose}
    >
      <img
        className="image-lightbox-image"
        src={src}
        alt={alt}
        onClick={(event) => event.stopPropagation()}
      />
    </div>
  );
}

export default ImageLightbox;
