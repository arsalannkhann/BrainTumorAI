import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileImage, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

export default function UploadDropzone({ onFileSelect, isProcessing }) {
    const onDrop = useCallback(acceptedFiles => {
        if (acceptedFiles?.length > 0) {
            onFileSelect(acceptedFiles[0]);
        }
    }, [onFileSelect]);

    const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': [],
            'image/png': [],
            'application/octet-stream': ['.npy']
        },
        disabled: isProcessing,
        maxFiles: 1
    });

    return (
        <div
            {...getRootProps()}
            className={clsx(
                "border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200",
                isDragActive ? "border-blue-500 bg-blue-50 scale-102" : "border-gray-300 hover:border-gray-400 hover:bg-gray-50",
                isProcessing && "opacity-50 cursor-not-allowed",
                isDragReject && "border-red-500 bg-red-50"
            )}
        >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center justify-center space-y-4">
                <div className={clsx(
                    "p-4 rounded-full",
                    isDragActive ? "bg-blue-100 text-blue-600" : "bg-gray-100 text-gray-500"
                )}>
                    {isDragReject ? <AlertCircle size={32} className="text-red-500" /> : <UploadCloud size={32} />}
                </div>

                <div className="space-y-1">
                    <p className="text-lg font-medium text-gray-900">
                        {isDragActive ? "Drop image here" : "Click or drag & drop"}
                    </p>
                    <p className="text-sm text-gray-500">
                        Supports .png, .jpg, .npy (MRI)
                    </p>
                </div>
            </div>
        </div>
    );
}
