import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, AlertCircle } from 'lucide-react';
import UploadDropzone from '../components/UploadDropzone';
import ResultsDashboard from '../components/ResultsDashboard';

export default function Analyze() {
    const [file, setFile] = useState(null);
    const [report, setReport] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState(null);

    const handleFileSelect = async (selectedFile) => {
        setFile(selectedFile);
        setError(null);
        setReport(null);
        setIsProcessing(true);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('run_classification', 'true');
        formData.append('run_segmentation', 'true');

        try {
            const response = await axios.post('/api/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setReport(response.data);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || "An unexpected error occurred during analysis.");
        } finally {
            setIsProcessing(false);
        }
    };

    const resetAnalysis = () => {
        setFile(null);
        setReport(null);
        setError(null);
    };

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">Brain Tumor Analysis</h1>
                <p className="mt-2 text-gray-600">
                    Upload an MRI scan to generate classification prediction and segmentation masks.
                </p>
            </div>

            {!report && (
                <div className="max-w-xl mx-auto">
                    <UploadDropzone
                        onFileSelect={handleFileSelect}
                        isProcessing={isProcessing}
                    />

                    {isProcessing && (
                        <div className="mt-8 text-center animate-in fade-in duration-500">
                            <div className="flex flex-col items-center justify-center space-y-3">
                                <Loader2 className="animate-spin text-blue-600" size={32} />
                                <p className="text-lg font-medium text-gray-900">Processing MRI Scan...</p>
                                <p className="text-sm text-gray-500">Running Deep Neural Networks</p>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
                            <AlertCircle className="text-red-500" size={20} />
                            <span className="text-red-700">{error}</span>
                        </div>
                    )}
                </div>
            )}

            {report && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <ResultsDashboard
                        report={report}
                        originalFile={file}
                        onReset={resetAnalysis}
                    />

                    <div className="mt-12 flex justify-center">
                        <button
                            onClick={resetAnalysis}
                            className="px-6 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors"
                        >
                            Analyze Another Image
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
