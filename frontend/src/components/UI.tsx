// src/components/ui.tsx
import React from 'react';
import { Plus, Minus } from 'lucide-react';

interface FormRowProps {
  label: string;
  children: React.ReactNode;
  htmlFor?: string;
  className?: string;
}

export const FormRow: React.FC<FormRowProps> = ({
  label,
  children,
  htmlFor,
  className = '',
}) => (
  <div
    className={`grid grid-cols-1 gap-2 md:grid-cols-3 md:gap-4 items-center ${className}`}
  >
    <Label htmlFor={htmlFor}>{label}</Label>
    <div className="md:col-span-2">{children}</div>
  </div>
);

export const Label: React.FC<{ children: React.ReactNode; htmlFor?: string }> = ({
  children,
  htmlFor,
}) => (
  <label
    htmlFor={htmlFor}
    className="text-sm font-medium text-gray-300 text-left"
  >
    {children}
  </label>
);

export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = (
  props
) => (
  <input
    {...props}
    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
  />
);

export const NumberInput: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = (
  props
) => (
  <input
    type="number"
    {...props}
    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
  />
);

export const Select: React.FC<React.SelectHTMLAttributes<HTMLSelectElement>> = (
  props
) => (
  <select
    {...props}
    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
  >
    {props.children}
  </select>
);

export const Checkbox: React.FC<{
  label: string;
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  id?: string;
}> = ({ label, checked, onChange, id }) => (
  <label htmlFor={id} className="flex items-center space-x-2 cursor-pointer">
    <input
      type="checkbox"
      id={id}
      checked={checked}
      onChange={onChange}
      className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
    />
    <span className="text-sm text-gray-300">{label}</span>
  </label>
);

interface ToggleButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  checked: boolean;
  children: React.ReactNode;
  variant?: 'blue' | 'red-green' | 'green-red';
  className?: string;
}

export const ToggleButton: React.FC<ToggleButtonProps> = ({
  checked,
  children,
  variant = 'blue',
  className = '',
  ...props
}) => {
  let styles =
    `w-full px-4 py-2 rounded-md font-medium transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 text-sm ${className}`;

  switch (variant) {
    case 'blue':
      styles += checked
        ? ' bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
        : ' bg-gray-700 text-gray-300 hover:bg-gray-600 focus:ring-gray-500';
      break;
    case 'red-green': // Not checked: Red, Checked: Green
      styles += checked
        ? ' bg-green-700 text-white hover:bg-green-600 focus:ring-green-500'
        : ' bg-red-700 text-white hover:bg-red-600 focus:ring-red-500';
      break;
    case 'green-red': // Not checked: Green, Checked: Red
      styles += checked
        ? ' bg-red-700 text-white hover:bg-red-600 focus:ring-red-500'
        : ' bg-green-700 text-white hover:bg-green-600 focus:ring-green-500';
      break;
  }

  return (
    <button type="button" aria-pressed={checked} {...props} className={styles}>
      {children}
    </button>
  );
};

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  children,
  className = "",
}) => {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div className={`col-span-1 md:col-span-3 ${className}`}>
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        className={`flex items-center justify-between w-full p-3 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
          isOpen
            ? 'bg-gray-700 rounded-b-none'
            : 'bg-transparent border border-gray-600 hover:bg-gray-800'
        }`}
      >
        <h3 className="text-md font-semibold text-white">{title}</h3>
        {isOpen ? (
          <Minus className="w-5 h-5 text-gray-300" />
        ) : (
          <Plus className="w-5 h-5 text-gray-300" />
        )}
      </button>

      {isOpen && (
        <div className="p-4 bg-gray-700 rounded-b-md space-y-4">
          {children}
        </div>
      )}
    </div>
  );
};

export const SectionTitle: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = '',
}) => (
  <h2 className={`text-lg font-semibold text-white mt-4 mb-3 col-span-1 md:col-span-3 ${className}`}>
    {children}
  </h2>
);