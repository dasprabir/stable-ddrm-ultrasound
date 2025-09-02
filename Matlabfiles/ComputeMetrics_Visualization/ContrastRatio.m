function medianCR = ContrastRatio(MatOut)

%Size of patchs
SZ = 4;
SX = 4;

PWTD = MatOut;
[Nz, Nx] = size(PWTD);


PWTD = PWTD(1:floor(Nz/SZ)*SZ,1+Nx-floor(Nx/SX)*SX:end); 
[Nz, Nx] = size(PWTD);

N11 = SZ*ones(1,Nz/SZ);
N22 = SX*ones(1,Nx/SX);
PWTD2 = mat2cell(PWTD,N11,N22);
PWTD2 = reshape(PWTD2,Nz/SZ*Nx/SX,[]);

%Define reference Patch
%R21 = PWTD(zz:zz+SZ-1,xx:xx+SX-1);  % fixed position
patchs=zeros(1,length(PWTD2));
for i=1:length(PWTD2)
R11 = PWTD2{i};
if IsPatchValid(R11)
patchs(i) = mean(R11(:).^2); 
else 
    patchs(i)=0;
end
end

patchs(patchs == 0 ) = NaN;
[A,I] = min(patchs);
R21 = PWTD2{I};
RefPower = mean(R21(:).^2);

% Calc CR
CR=zeros(1,length(PWTD2));
for i=1:length(PWTD2)
R11 = PWTD2{i};
CR(i) = 10*log10(mean(R11(:).^2) / RefPower);  %  = mean(R21(:).^2)
end
CR = CR(isfinite(CR));
CR = CR(CR>0);

medianCR = median(CR,'all');

% show boxplot
listis= boxplot(CR, 'Whisker', 2);
CRis=get(listis(6)).YData;
ylabel('CR [dB]')
%fprintf('CR: %f\n',[CRis(1)]);

% show reference patch
% figure 
% imagesc(R21)

end

function isValid = IsPatchValid(InMat)
    % IsPatchValid checks if all quadrants of InMat contain at least one non-zero element.
    %
    % Input:
    %   InMat - The input matrix.
    %
    % Output:
    %   isValid - Returns true if all quadrants have at least one non-zero element; false otherwise.

    % Get the dimensions of the matrix
    [rows, cols] = size(InMat);

    % Find the midpoint indices for rows and columns
    midRow = ceil(rows / 2);
    midCol = ceil(cols / 2);

    % Define the four quadrants
    Q1 = InMat(1:midRow, 1:midCol);           % Top-left quadrant
    Q2 = InMat(1:midRow, midCol+1:end);       % Top-right quadrant
    Q3 = InMat(midRow+1:end, 1:midCol);       % Bottom-left quadrant
    Q4 = InMat(midRow+1:end, midCol+1:end);   % Bottom-right quadrant

    % Check if each quadrant contains at least one non-zero element
    isValid = any(Q1(:)) && any(Q2(:)) && any(Q3(:)) && any(Q4(:));
end

