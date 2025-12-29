import React, { useState } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { Eye, Layers, Brain, CheckCircle, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

export default function ResultsDashboard({ report }) {
    const [viewMode, setViewMode] = useState('original'); // original, gradcam, segmentation
    const [showMask, setShowMask] = useState(true);

    const { classification, segmentation, validation, image_path, gradcam_path } = report;

    // Format probability data
    const probData = {
        labels: ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'],
        datasets: [
            {
                label: 'Confidence',
                data: [
                    classification?.class_probabilities.glioma * 100,
                    classification?.class_probabilities.meningioma * 100,
                    classification?.class_probabilities.pituitary * 100,
                    classification?.class_probabilities.no_tumor * 100,
                ],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(107, 114, 128, 0.7)',
                ],
                borderRadius: 4,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                ticks: {
                    callback: (value) => value + '%'
                }
            }
        },
        plugins: {
            legend: { display: false },
        },
    };

    // Resolve Image URLs
    // In dev mode, /data maps to backend /data via proxy
    const originalImageUrl = `/` + image_path;
    const gradcamUrl = gradcam_path ? `/` + gradcam_path : null;
    const maskUrl = segmentation?.mask_path ? `/` + segmentation.mask_path : null;

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column: Visualizations */}
            <div className="space-y-6">
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                    <div className="p-4 border-b border-gray-200 flex justify-between items-center bg-gray-50">
                        <h3 className="font-semibold text-gray-900">Image Visualization</h3>
                        <div className="flex space-x-2">
                            <button
                                onClick={() => setViewMode('original')}
                                className={clsx(
                                    "p-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1",
                                    viewMode === 'original' ? "bg-white shadow text-blue-600" : "text-gray-500 hover:bg-gray-100"
                                )}
                            >
                                <Eye size={16} />
                                <span>Original</span>
                            </button>
                            {gradcamUrl && (
                                <button
                                    onClick={() => setViewMode('gradcam')}
                                    className={clsx(
                                        "p-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1",
                                        viewMode === 'gradcam' ? "bg-white shadow text-purple-600" : "text-gray-500 hover:bg-gray-100"
                                    )}
                                >
                                    <Brain size={16} />
                                    <span>Grad-CAM</span>
                                </button>
                            )}
                            {maskUrl && (
                                <button
                                    onClick={() => {
                                        setViewMode('segmentation');
                                        setShowMask(true);
                                    }}
                                    className={clsx(
                                        "p-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1",
                                        viewMode === 'segmentation' ? "bg-white shadow text-green-600" : "text-gray-500 hover:bg-gray-100"
                                    )}
                                >
                                    <Layers size={16} />
                                    <span>Segment</span>
                                </button>
                            )}
                        </div>
                    </div>

                    <div className="relative aspect-square bg-black flex items-center justify-center overflow-hidden">

                        {/* Base Image */}
                        {viewMode === 'original' && (
                            <img
                                src={originalImageUrl}
                                alt="MRI Scan"
                                className="w-full h-full object-contain"
                                onError={(e) => { e.target.src = "https://placehold.co/600x600?text=Image+Load+Error"; }}
                            />
                        )}

                        {/* Grad-CAM View */}
                        {viewMode === 'gradcam' && gradcamUrl && (
                            <img
                                src={gradcamUrl}
                                alt="Explainability Heatmap"
                                className="w-full h-full object-contain"
                            />
                        )}

                        {/* Segmentation View */}
                        {viewMode === 'segmentation' && (
                            <div className="relative w-full h-full">
                                <img
                                    src={originalImageUrl}
                                    alt="Original for segmentation"
                                    className="absolute inset-0 w-full h-full object-contain"
                                />
                                {showMask && maskUrl && (
                                    <img
                                        src={maskUrl}
                                        alt="Tumor Mask"
                                        className="absolute inset-0 w-full h-full object-contain opacity-60 mix-blend-normal"
                                    />
                                )}
                                {!maskUrl && (
                                    <div className="absolute top-4 left-4 bg-black/70 text-white p-2 text-xs rounded">
                                        Mask visualization not available.
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Right Column: Stats & Probabilities */}
            <div className="space-y-6">

                {/* Classification Result */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                    <div className="flex justify-between items-start">
                        <div>
                            <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Prediction</h2>
                            <div className="mt-1 flex items-baseline space-x-3">
                                <span className={clsx(
                                    "text-3xl font-bold",
                                    classification?.predicted_class === "No Tumor" ? "text-gray-900" : "text-red-600"
                                )}>
                                    {classification?.predicted_class}
                                </span>
                                <span className="text-lg text-gray-500 font-medium">
                                    {(classification?.confidence_score * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>
                        {validation.requires_manual_review ? (
                            <div className="flex items-center space-x-1 bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-medium">
                                <AlertTriangle size={16} />
                                <span>Review Needed</span>
                            </div>
                        ) : (
                            <div className="flex items-center space-x-1 bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                                <CheckCircle size={16} />
                                <span>High Confidence</span>
                            </div>
                        )}
                    </div>

                    <div className="mt-6 h-64">
                        <Bar data={probData} options={chartOptions} />
                    </div>
                </div>

                {/* Segmentation Stats */}
                {segmentation && (
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                        <h3 className="font-semibold text-gray-900 border-b pb-2 mb-4">Segmentation Analysis</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gray-50 p-3 rounded-lg">
                                <p className="text-xs text-gray-500 uppercase">Tumor Area</p>
                                <p className="text-xl font-bold text-gray-900">{segmentation.stats.tumor_area_percentage.toFixed(2)}%</p>
                            </div>
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-gray-600">Edema</span>
                                    <span className={segmentation.stats.edema_present ? "text-red-600 font-bold" : "text-gray-400"}>
                                        {segmentation.stats.edema_present ? "Detected" : "None"}
                                    </span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-gray-600">Enhancing</span>
                                    <span className={segmentation.stats.enhancing_present ? "text-blue-600 font-bold" : "text-gray-400"}>
                                        {segmentation.stats.enhancing_present ? "Detected" : "None"}
                                    </span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-gray-600">Necrotic</span>
                                    <span className={segmentation.stats.necrotic_present ? "text-green-600 font-bold" : "text-gray-400"}>
                                        {segmentation.stats.necrotic_present ? "Detected" : "None"}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
}
