import React from 'react';
import { Eye, Clock } from 'lucide-react';

export default function HistoryTable({ records }) {
    if (!records || records.length === 0) {
        return (
            <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
                <Clock className="mx-auto text-gray-400 mb-4" size={48} />
                <h3 className="text-lg font-medium text-gray-900">No History Found</h3>
                <p className="text-gray-500">Run an analysis to see specific history.</p>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                    <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            ID / Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Prediction
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Confidence
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Findings
                        </th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Image
                        </th>
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {records.map((record) => (
                        <tr key={record.id} className="hover:bg-gray-50 transition-colors">
                            <td className="px-6 py-4 whitespace-nowrap">
                                <div className="text-sm font-medium text-gray-900">#{record.id}</div>
                                <div className="text-sm text-gray-500">
                                    {new Date(record.timestamp).toLocaleDateString()}
                                </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                    {record.classification.predicted_class}
                                </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {(record.classification.confidence_score * 100).toFixed(1)}%
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {record.segmentation ? (
                                    <span>Tumor: {record.segmentation.stats.tumor_area_percentage.toFixed(1)}%</span>
                                ) : "N/A"}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                {record.image_path && (
                                    <img
                                        src={`/${record.image_path}`}
                                        className="h-10 w-10 object-cover rounded inline-block border border-gray-200"
                                        alt="Thumbnail"
                                    />
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
