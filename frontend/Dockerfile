# Use Node.js LTS version as base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy rest of the application code
COPY . .

# Build the Next.js application
RUN npm run build

# Expose port 8080 to match docker-compose configuration
EXPOSE 8080

# Set the environment variable for the port
ENV PORT=8080

# Start the application
CMD ["npm", "start"]
