import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Brain, ShieldCheck, Activity } from 'lucide-react';

export default function Home() {
    return (
        <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
            <div className="sm:mx-auto sm:w-full sm:max-w-4xl">
                <div className="text-center">
                    <Brain className="mx-auto h-24 w-24 text-blue-600" />
                    <h2 className="mt-6 text-4xl font-extrabold text-gray-900 tracking-tight">
                        Brain Tumor AI Diagnostics
                    </h2>
                    <p className="mt-4 text-xl text-gray-600">
                        Advanced deep learning system for tumor classification and segmentation.
                        research-grade accuracy with explainable AI.
                    </p>
                </div>

                <div className="mt-12 grid gap-8 md:grid-cols-3">
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 text-center">
                        <div className="mx-auto w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                            <Activity className="text-blue-600" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-900">Classification</h3>
                        <p className="mt-2 text-gray-500">
                            Detects Glioma, Meningioma, and Pituitary tumors with high confidence.
                        </p>
                    </div>

                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 text-center">
                        <div className="mx-auto w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mb-4">
                            <Brain className="text-green-600" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-900">Segmentation</h3>
                        <p className="mt-2 text-gray-500">
                            Precise voxel-level segmentation of edema, enhancing tumor, and necrotic core.
                        </p>
                    </div>

                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 text-center">
                        <div className="mx-auto w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mb-4">
                            <ShieldCheck className="text-purple-600" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-900">Explainable</h3>
                        <p className="mt-2 text-gray-500">
                            Grad-CAM visualizations to understand model focus and build trust.
                        </p>
                    </div>
                </div>

                <div className="mt-12 text-center">
                    <Link
                        to="/analyze"
                        className="inline-flex items-center px-8 py-3 border border-transparent text-lg font-medium rounded-full shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all transform hover:scale-105"
                    >
                        Start Analysis
                        <ArrowRight className="ml-2 -mr-1 h-5 w-5" />
                    </Link>
                </div>
            </div>
        </div>
    );
}
