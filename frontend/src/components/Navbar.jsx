import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Brain, History, Activity } from 'lucide-react';
import clsx from 'clsx';

export default function Navbar() {
    const location = useLocation();

    const NavItem = ({ to, icon: Icon, label }) => {
        const isActive = location.pathname === to;
        return (
            <Link
                to={to}
                className={clsx(
                    "flex items-center space-x-2 px-4 py-2 rounded-md transition-colors",
                    isActive
                        ? "bg-blue-50 text-blue-600 font-medium"
                        : "text-gray-600 hover:bg-gray-100"
                )}
            >
                <Icon size={20} />
                <span>{label}</span>
            </Link>
        );
    };

    return (
        <nav className="bg-white border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    <div className="flex items-center">
                        <Link to="/" className="flex items-center space-x-2">
                            <div className="bg-blue-600 p-2 rounded-lg">
                                <Brain className="text-white" size={24} />
                            </div>
                            <span className="text-xl font-bold text-gray-900">NeuroAI</span>
                        </Link>
                    </div>

                    <div className="flex items-center space-x-4">
                        <NavItem to="/" icon={Activity} label="Analysis" />
                        <NavItem to="/history" icon={History} label="History" />
                    </div>
                </div>
            </div>
        </nav>
    );
}
