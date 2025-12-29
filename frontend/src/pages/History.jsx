import React, { useEffect, useState } from 'react';
import axios from 'axios';
import HistoryTable from '../components/HistoryTable';
import { Loader2 } from 'lucide-react';

export default function History() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        try {
            const response = await axios.get('/api/history');
            setData(response.data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">Analysis History</h1>
                <p className="mt-2 text-gray-600">
                    View past predictions and patient reports.
                </p>
            </div>

            {loading ? (
                <div className="flex justify-center p-12">
                    <Loader2 className="animate-spin text-blue-600" size={32} />
                </div>
            ) : (
                <HistoryTable records={data} />
            )}
        </div>
    );
}
